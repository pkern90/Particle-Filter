#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = 50;
    weights.resize((unsigned long) num_particles);
    particles.resize((unsigned long) num_particles);

    random_device rd;
    default_random_engine gen(rd());
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {

        struct Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.d;

        particles[i] = p;
        weights[i] = (p.weight);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

    for (int i = 0; i < num_particles; i++) {

        Particle *p = &particles[i];
        double x_new = p->x + velocity / yaw_rate * (sin(p->theta + yaw_rate * delta_t) - sin(p->theta));
        double y_new = p->y + velocity / yaw_rate * (cos(p->theta) - cos(p->theta + yaw_rate * delta_t));
        double theta_new = p->theta + yaw_rate * delta_t;

        random_device rd;
        default_random_engine gen(rd());
        normal_distribution<double> dist_x(x_new, std_pos[0]);
        normal_distribution<double> dist_y(y_new, std_pos[1]);
        normal_distribution<double> dist_theta(theta_new, std_pos[2]);

        p->x = dist_x(gen);
        p->y = dist_y(gen);
        p->theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs> &observations) {

}

LandmarkObs align_space(Particle particle, LandmarkObs observation) {
    LandmarkObs transformed_obs;

    transformed_obs.id = observation.id;
    transformed_obs.x = particle.x + (observation.x * cos(particle.theta)) - (observation.y * sin(particle.theta));
    transformed_obs.y = particle.y + (observation.x * sin(particle.theta)) + (observation.y * cos(particle.theta));

    return transformed_obs;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   vector<LandmarkObs> observations, Map map_landmarks) {

    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    double sum_weights = 0.;

    for (int p_i = 0; p_i < num_particles; p_i++) {
        Particle *p = &particles[p_i];
        double weight = 1.;

        for (int obs_i = 0; obs_i < observations.size(); obs_i++) {
            LandmarkObs obs = observations[obs_i];
            obs = align_space(*p, obs);

            Map::single_landmark_s best_lm;
            double shortest_dist = numeric_limits<double>::max();
            for (int lm_i = 0; lm_i < map_landmarks.landmark_list.size(); lm_i++) {
                Map::single_landmark_s lm = map_landmarks.landmark_list[lm_i];
                double cur_dist = dist(obs.x, obs.y, lm.x_f, lm.y_f);
                if (cur_dist < shortest_dist) {
                    shortest_dist = cur_dist;
                    best_lm = lm;
                }
            }

            double numerator = exp(-0.5 *
                                   (pow((obs.x - best_lm.x_f), 2) / pow(std_x, 2) +
                                    pow((obs.y - best_lm.y_f), 2) / pow(std_y, 2)));
            double denominator = 2 * M_PI * std_x * std_y;
            weight *= numerator / denominator;
        }

        sum_weights += weight;
        p->weight = weight;
    }

    for (int i = 0; i < num_particles; i++) {
        Particle *p = &particles[i];
        p->weight /= sum_weights;
        weights[i] = p->weight;
    }
}

void ParticleFilter::resample() {
    random_device rd;
    default_random_engine gen(rd());

    discrete_distribution<> dist_particles(weights.begin(), weights.end());
    vector<Particle> resampled_particles((unsigned long) num_particles);

    for (int i = 0; i < num_particles; i++) {
        resampled_particles[i] = particles[dist_particles(gen)];
    }

    particles = resampled_particles;
}

void ParticleFilter::write(string filename) {
    // You don't need to modify this file.
    ofstream dataFile;
    dataFile.open(filename, ios::app);
    for (int i = 0; i < num_particles; ++i) {
        dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
    }
    dataFile.close();
}
