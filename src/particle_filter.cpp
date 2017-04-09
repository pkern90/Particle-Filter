#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 50;

	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	for(int i = 0; i < num_particles; i++){

		struct Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.d;

		particles.push_back(p);
        weights.push_back(p.weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

    for(int i = 0; i < num_particles; i++){

        Particle* p = &particles[i];
        double x_new = p->x + velocity/yaw_rate * (sin(p->theta + yaw_rate * delta_t) - sin(p->theta));
        double y_new = p->y + velocity/yaw_rate * (cos(p->theta) - cos(p->theta + yaw_rate * delta_t));
        double theta_new = p->theta + yaw_rate * delta_t;

        std::default_random_engine gen;
        std::normal_distribution<double> dist_x(x_new, std_pos[0]);
        std::normal_distribution<double> dist_y(y_new, std_pos[1]);
        std::normal_distribution<double> dist_theta(theta_new, std_pos[2]);

        p->x = dist_x(gen);
        p->y = dist_y(gen);
        p->theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
    std::default_random_engine gen;
    std::discrete_distribution<> dist_particles(weights.begin(), weights.end());
    std::vector<Particle> resampled_particles;

    for(int i = 0; i < num_particles; i++){
        resampled_particles.push_back(particles[dist_particles(gen)]);
    }

    particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
