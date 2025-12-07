
# Generated: 2025-08-27T16:17:15.230131
# Source Brief: brief_01180.md
# Brief Index: 1180

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to accelerate, ←→ to turn, ↓ to brake. Hold Shift to drift and press Space to activate your boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Drift through corners, manage your boost, and race against opponents on procedurally generated tracks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500 # Increased for longer races
        self.LAPS_TO_WIN = 2
        self.MAX_COLLISIONS = 4
        
        # Colors
        self.COLOR_BG = (30, 35, 40)
        self.COLOR_TRACK = (80, 80, 90)
        self.COLOR_TRACK_BORDER = (120, 120, 130)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_OPPONENTS = [(255, 50, 50), (50, 150, 255), (255, 255, 50)]
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_BOOST_METER = (0, 200, 255)
        self.COLOR_BOOST_PARTICLE = (0, 220, 255)
        self.COLOR_DRIFT_PARTICLE = (200, 200, 200)

        # Exact spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans-serif", 18)
        self.font_large = pygame.font.SysFont("sans-serif", 36)

        # Initialize state variables
        self.player = {}
        self.opponents = []
        self.track_center = []
        self.track_inner = []
        self.track_outer = []
        self.checkpoints = []
        self.particles = []
        self.camera_pos = np.array([0.0, 0.0])

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""

        self._generate_track()
        self._spawn_player()
        self._spawn_opponents()

        self.particles = []
        self.camera_pos = np.copy(self.player['pos'])

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        # Update game logic
        reward += self._update_player(movement, space_held, shift_held)
        self._update_opponents()
        self._update_particles()
        
        # Calculate rewards from events
        collision_reward = self._handle_collisions()
        lap_reward, lap_completed = self._check_lap_completion()
        reward += collision_reward
        reward += lap_reward
        
        if lap_completed:
            self.opponent_base_speed += 0.5 # Increase difficulty

        # Time penalty
        reward -= 0.02
        self.score += reward

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.win_message:
            if self.player['laps'] >= self.LAPS_TO_WIN:
                self.win_message = "YOU WIN!"
                reward += 50
            elif self.player['collisions'] >= self.MAX_COLLISIONS:
                self.win_message = "TOO MANY COLLISIONS"
                reward -= 50
            else:
                self.win_message = "TIME UP"
        
        self.score += reward if terminated else 0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_track(self):
        center = np.array([0, 0])
        num_points = 12
        min_radius, max_radius = 250, 400
        
        # Generate random waypoints
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            radius = self.np_random.uniform(min_radius, max_radius)
            points.append(center + np.array([math.cos(angle), math.sin(angle)]) * radius)
        
        # Create a smooth Catmull-Rom spline for the centerline
        self.track_center = []
        for i in range(num_points):
            p0 = points[(i - 1 + num_points) % num_points]
            p1 = points[i]
            p2 = points[(i + 1) % num_points]
            p3 = points[(i + 2) % num_points]
            for t in np.linspace(0, 1, 20, endpoint=False):
                point = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3)
                self.track_center.append(point)

        # Create track boundaries and checkpoints
        self.track_inner, self.track_outer, self.checkpoints = [], [], []
        track_width = 80
        for i in range(len(self.track_center)):
            p1 = self.track_center[i]
            p2 = self.track_center[(i + 1) % len(self.track_center)]
            tangent = p2 - p1
            if np.linalg.norm(tangent) > 0:
                normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
                self.track_outer.append(p1 + normal * track_width / 2)
                self.track_inner.append(p1 - normal * track_width / 2)
            
            if i % 20 == 0: # Place a checkpoint every 20 segments
                self.checkpoints.append({'pos': p1, 'radius': track_width, 'id': len(self.checkpoints)})

    def _spawn_player(self):
        start_pos = self.track_center[0]
        start_dir = self.track_center[1] - self.track_center[0]
        self.player = {
            'pos': np.copy(start_pos),
            'vel': np.array([0.0, 0.0]),
            'angle': math.atan2(start_dir[1], start_dir[0]),
            'radius': 12,
            'boost': 100.0,
            'max_boost': 100.0,
            'laps': 0,
            'collisions': 0,
            'checkpoints_hit': set(),
            'last_checkpoint': -1
        }

    def _spawn_opponents(self):
        self.opponents = []
        self.opponent_base_speed = 2.5
        num_opponents = 3
        for i in range(num_opponents):
            start_index = (len(self.track_center) // (num_opponents + 1)) * (i + 1)
            start_pos = self.track_center[start_index]
            self.opponents.append({
                'pos': np.copy(start_pos),
                'radius': 12,
                'color': self.COLOR_OPPONENTS[i % len(self.COLOR_OPPONENTS)],
                'target_waypoint_idx': start_index
            })

    def _update_player(self, movement, space_held, shift_held):
        # Physics params
        turn_speed = 0.08
        acceleration = 0.4
        max_speed = 5.0
        brake_force = 0.9
        friction = 0.96
        drift_friction = 0.985
        drift_turn_mod = 1.5

        # Drifting
        is_drifting = shift_held and np.linalg.norm(self.player['vel']) > 2.0
        if is_drifting:
            turn_speed *= drift_turn_mod
            friction = drift_friction
            if self.steps % 3 == 0:
                self._add_particle(self.player['pos'], self.COLOR_DRIFT_PARTICLE, 20, 1.5, self.player['angle'] + math.pi)

        # Steering
        if movement == 3: self.player['angle'] -= turn_speed
        if movement == 4: self.player['angle'] += turn_speed

        # Acceleration/Braking
        forward_vec = np.array([math.cos(self.player['angle']), math.sin(self.player['angle'])])
        if movement == 1: # Accelerate
            self.player['vel'] += forward_vec * acceleration
        elif movement == 2: # Brake
            self.player['vel'] *= brake_force

        # Boosting
        if space_held and self.player['boost'] > 0:
            self.player['vel'] += forward_vec * (acceleration * 1.5)
            self.player['boost'] = max(0, self.player['boost'] - 1.5)
            if self.steps % 2 == 0: # Add boost particles
                self._add_particle(self.player['pos'], self.COLOR_BOOST_PARTICLE, 30, 2.5, self.player['angle'] + math.pi + self.np_random.uniform(-0.2, 0.2))
        else:
             self.player['boost'] = min(self.player['max_boost'], self.player['boost'] + 0.2)

        # Apply friction and cap speed
        self.player['vel'] *= friction
        speed = np.linalg.norm(self.player['vel'])
        if speed > max_speed:
            self.player['vel'] = self.player['vel'] / speed * max_speed
        
        # Update position
        self.player['pos'] += self.player['vel']
        
        # Reward for moving forward
        forward_movement = np.dot(self.player['vel'], forward_vec)
        return max(0, forward_movement * 0.02)

    def _update_opponents(self):
        for opp in self.opponents:
            target_pos = self.track_center[opp['target_waypoint_idx']]
            dist_to_target = np.linalg.norm(target_pos - opp['pos'])

            if dist_to_target < 30: # Close enough, switch to next waypoint
                opp['target_waypoint_idx'] = (opp['target_waypoint_idx'] + 1) % len(self.track_center)
                target_pos = self.track_center[opp['target_waypoint_idx']]
            
            move_dir = target_pos - opp['pos']
            if np.linalg.norm(move_dir) > 0:
                move_dir /= np.linalg.norm(move_dir)
            
            opp['pos'] += move_dir * self.opponent_base_speed

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.95

    def _add_particle(self, pos, color, life, speed, angle):
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed * self.np_random.uniform(0.5, 1.5)
        self.particles.append({
            'pos': np.copy(pos),
            'vel': vel,
            'life': life,
            'max_life': life,
            'color': color,
            'radius': self.np_random.uniform(2, 5)
        })

    def _handle_collisions(self):
        reward = 0
        # Player-Opponent
        for opp in self.opponents:
            dist = np.linalg.norm(self.player['pos'] - opp['pos'])
            if dist < self.player['radius'] + opp['radius']:
                self.player['collisions'] += 1
                reward -= 1.0
                # Simple physics response
                self.player['vel'] *= -0.5
                # Reset to track to prevent getting stuck
                self.player['pos'] = self._get_closest_track_point(self.player['pos'])

        # Player-Wall
        if not self._is_on_track(self.player['pos']):
            self.player['vel'] *= -0.8 # Bounce off wall
            self.player['pos'] = self._get_closest_track_point(self.player['pos'])
        
        return reward

    def _get_closest_track_point(self, pos):
        dists = [np.linalg.norm(pos - p) for p in self.track_center]
        return self.track_center[np.argmin(dists)]
    
    def _is_on_track(self, pos):
        # A simplified check using distance to centerline
        dists = [np.linalg.norm(pos - p) for p in self.track_center]
        return min(dists) < 40

    def _check_lap_completion(self):
        reward = 0
        lap_completed = False

        # Checkpoint logic
        for cp in self.checkpoints:
            dist = np.linalg.norm(self.player['pos'] - cp['pos'])
            if dist < cp['radius'] and cp['id'] not in self.player['checkpoints_hit']:
                # Must hit checkpoints in order
                if cp['id'] == (self.player['last_checkpoint'] + 1) % len(self.checkpoints):
                    self.player['checkpoints_hit'].add(cp['id'])
                    self.player['last_checkpoint'] = cp['id']
        
        # Finish line logic
        finish_line_pos = self.track_center[0]
        dist_to_finish = np.linalg.norm(self.player['pos'] - finish_line_pos)
        
        if dist_to_finish < 40 and len(self.player['checkpoints_hit']) == len(self.checkpoints):
            self.player['laps'] += 1
            self.player['checkpoints_hit'] = set() # Reset for next lap
            self.player['last_checkpoint'] = -1
            reward += 5.0
            lap_completed = True

        return reward, lap_completed

    def _check_termination(self):
        self.game_over = (
            self.player['laps'] >= self.LAPS_TO_WIN or
            self.player['collisions'] >= self.MAX_COLLISIONS or
            self.steps >= self.MAX_STEPS
        )
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.player.get('laps', 0),
            "collisions": self.player.get('collisions', 0),
            "boost": self.player.get('boost', 0)
        }

    def _world_to_screen(self, pos):
        # Isometric projection + camera offset
        iso_x = (pos[0] - pos[1]) * 0.8
        iso_y = (pos[0] + pos[1]) * 0.4
        screen_x = iso_x - self.camera_pos[0] + self.WIDTH / 2
        screen_y = iso_y - self.camera_pos[1] + self.HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _update_camera(self):
        # Camera follows player with a lead based on velocity
        lead_factor = 0.1
        target_world_pos = self.player['pos'] + self.player['vel'] * 5
        target_iso_x = (target_world_pos[0] - target_world_pos[1]) * 0.8
        target_iso_y = (target_world_pos[0] + target_world_pos[1]) * 0.4
        
        camera_target = np.array([target_iso_x, target_iso_y])
        self.camera_pos += (camera_target - self.camera_pos) * lead_factor
    
    def _render_game(self):
        self._update_camera()
        
        # Render track
        track_poly_outer = [self._world_to_screen(p) for p in self.track_outer]
        track_poly_inner = [self._world_to_screen(p) for p in self.track_inner]
        pygame.gfxdraw.filled_polygon(self.screen, track_poly_outer, self.COLOR_TRACK_BORDER)
        pygame.gfxdraw.filled_polygon(self.screen, track_poly_inner, self.COLOR_BG)
        pygame.gfxdraw.aapolygon(self.screen, track_poly_outer, self.COLOR_TRACK_BORDER)
        pygame.gfxdraw.aapolygon(self.screen, track_poly_inner, self.COLOR_TRACK_BORDER)

        # Render start/finish line
        p1 = self._world_to_screen(self.track_inner[0])
        p2 = self._world_to_screen(self.track_outer[0])
        pygame.draw.line(self.screen, (255, 255, 255), p1, p2, 5)
        
        # Render particles
        for p in self.particles:
            pos = self._world_to_screen(p['pos'])
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)

        # Render opponents
        for opp in self.opponents:
            pos = self._world_to_screen(opp['pos'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(opp['radius']), opp['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(opp['radius']), (255,255,255))

        # Render player
        player_pos_screen = self._world_to_screen(self.player['pos'])
        pygame.gfxdraw.filled_circle(self.screen, player_pos_screen[0], player_pos_screen[1], int(self.player['radius']), self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_pos_screen[0], player_pos_screen[1], int(self.player['radius']), (255,255,255))
        # Player direction indicator
        forward_vec = np.array([math.cos(self.player['angle']), math.sin(self.player['angle'])])
        indicator_end_world = self.player['pos'] + forward_vec * self.player['radius'] * 1.5
        indicator_end_screen = self._world_to_screen(indicator_end_world)
        pygame.draw.line(self.screen, (255, 255, 255), player_pos_screen, indicator_end_screen, 2)


    def _render_ui(self):
        # Lap counter
        lap_text = self.font_small.render(f"LAP: {self.player['laps'] + 1} / {self.LAPS_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(lap_text, (10, 10))

        # Collision counter
        col_text = self.font_small.render(f"COLLISIONS: {self.player['collisions']} / {self.MAX_COLLISIONS}", True, self.COLOR_TEXT)
        self.screen.blit(col_text, (self.WIDTH - col_text.get_width() - 10, 10))

        # Boost meter
        boost_w, boost_h = 150, 20
        boost_x, boost_y = self.WIDTH - boost_w - 10, self.HEIGHT - boost_h - 10
        boost_pct = self.player['boost'] / self.player['max_boost']
        pygame.draw.rect(self.screen, (50,50,50), (boost_x, boost_y, boost_w, boost_h))
        pygame.draw.rect(self.screen, self.COLOR_BOOST_METER, (boost_x, boost_y, int(boost_w * boost_pct), boost_h))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (boost_x, boost_y, boost_w, boost_h), 1)

        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            end_text = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Racer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Laps: {info['laps']}, Collisions: {info['collisions']}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()