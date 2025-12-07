
# Generated: 2025-08-28T04:16:21.120850
# Source Brief: brief_05190.md
# Brief Index: 5190

        
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

    user_guide = (
        "Controls: Arrow keys to move your ship. Hold space near an asteroid to mine it for ore."
    )

    game_description = (
        "Pilot a spaceship through a dense asteroid field. Mine asteroids for valuable ore to "
        "reach the target score, but be careful! Colliding with an asteroid will destroy your ship. "
        "Race against the clock to secure your fortune."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.fps = 30
        self.time_limit_seconds = 60
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # --- Visuals & Fonts ---
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 72)
        self.COLOR_BG = (10, 10, 26)
        self.COLOR_PLAYER = (0, 170, 255)
        self.COLOR_THRUST = (255, 120, 30)
        self.COLOR_ASTEROID = (128, 128, 140)
        self.COLOR_ORE = (255, 255, 0)
        self.COLOR_BEAM = (255, 255, 255, 150)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TIMER_OK = (0, 255, 128)
        self.COLOR_TIMER_WARN = (255, 255, 0)
        self.COLOR_TIMER_CRIT = (255, 0, 0)

        # --- Game Constants ---
        self.PLAYER_ACCEL = 0.4
        self.PLAYER_FRICTION = 0.96
        self.PLAYER_TURN_SPEED = 0.15
        self.PLAYER_RADIUS = 10
        self.WIN_SCORE = 50
        self.MAX_STEPS = self.time_limit_seconds * self.fps
        self.MINING_RANGE = 50
        self.MINING_RATE = 0.1 # Ore per frame
        self.INITIAL_ASTEROIDS = 25
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.time_remaining = self.MAX_STEPS

        self.player_pos = np.array([self.width / 2, self.height / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.player_angle = -math.pi / 2 # Pointing up

        self.is_mining = False
        self.mining_target = None
        
        self.asteroids = self._create_asteroids()
        self.stars = self._create_stars()
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.fps)
            
        reward = 0
        terminated = False
        
        if not self.game_over:
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            
            # --- Reward for movement towards/away from nearest asteroid ---
            prev_dist_to_nearest = self._get_dist_to_nearest_asteroid()
            
            self._handle_input(movement)
            self._update_player()
            
            new_dist_to_nearest = self._get_dist_to_nearest_asteroid()
            if new_dist_to_nearest is not None and prev_dist_to_nearest is not None:
                if new_dist_to_nearest < prev_dist_to_nearest:
                    reward += 0.01 # Small incentive to move towards asteroids
                else:
                    reward -= 0.01

            # --- Mining ---
            self.is_mining, mining_reward = self._handle_mining(space_held)
            reward += mining_reward
            
            # --- Update non-player entities ---
            self._update_asteroids()
            self._update_particles()
            
            # --- Collision Check ---
            collision_reward, collision_detected = self._check_collisions()
            reward += collision_reward
            if collision_detected:
                self.game_over = True
                terminated = True
                # sfx: explosion_large

            # --- Update Timers & Termination ---
            self.steps += 1
            self.time_remaining -= 1

            if self.score >= self.WIN_SCORE:
                self.game_over = True
                self.game_won = True
                terminated = True
                reward += 50
            
            if self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
                self.game_over = True
                terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        thrust = np.array([0.0, 0.0])
        target_angle = self.player_angle

        if movement == 1: # Up
            thrust[1] = -self.PLAYER_ACCEL
            target_angle = -math.pi / 2
        elif movement == 2: # Down
            thrust[1] = self.PLAYER_ACCEL
            target_angle = math.pi / 2
        elif movement == 3: # Left
            thrust[0] = -self.PLAYER_ACCEL
            target_angle = math.pi
        elif movement == 4: # Right
            thrust[0] = self.PLAYER_ACCEL
            target_angle = 0
        
        self.player_vel += thrust
        
        # Smoothly turn the ship
        angle_diff = (target_angle - self.player_angle + math.pi) % (2 * math.pi) - math.pi
        self.player_angle += np.clip(angle_diff, -self.PLAYER_TURN_SPEED, self.PLAYER_TURN_SPEED)
        
        if np.linalg.norm(thrust) > 0:
            self._create_thrust_particles()
            # sfx: ship_thrust_loop

    def _update_player(self):
        self.player_vel *= self.PLAYER_FRICTION
        self.player_pos += self.player_vel
        
        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.width - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.height - self.PLAYER_RADIUS)

    def _handle_mining(self, space_held):
        reward = 0
        is_mining = False
        self.mining_target = None
        
        if space_held:
            # Find closest asteroid within mining range
            closest_asteroid = None
            min_dist = self.MINING_RANGE
            for asteroid in self.asteroids:
                dist = np.linalg.norm(self.player_pos - asteroid['pos']) - asteroid['radius']
                if dist < min_dist:
                    min_dist = dist
                    closest_asteroid = asteroid
            
            if closest_asteroid:
                is_mining = True
                self.mining_target = closest_asteroid
                
                ore_mined_this_frame = min(self.MINING_RATE, closest_asteroid['ore'])
                closest_asteroid['ore'] -= ore_mined_this_frame
                self.score += ore_mined_this_frame
                reward += ore_mined_this_frame * 1.0 # +1 per ore unit
                
                # sfx: mining_beam_loop
                
                # Visual feedback for mining
                if self.np_random.random() < 0.5:
                    self._create_ore_particles(closest_asteroid)
                
                if closest_asteroid['ore'] <= 0:
                    self.asteroids.remove(closest_asteroid)
                    # sfx: asteroid_depleted
        
        return is_mining, reward

    def _check_collisions(self):
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_RADIUS + asteroid['radius']:
                # Collision occurred
                size_rewards = {15: -5, 25: -10, 40: -15}
                return size_rewards.get(asteroid['base_radius'], -10), True
        return 0, False

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['angle'] += asteroid['rot_speed']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": int(self.time_remaining / self.fps)}

    def _render_game(self):
        # Draw stars
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['size'])
            
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Draw mining beam
        if self.is_mining and self.mining_target:
            start_pos = tuple(self.player_pos.astype(int))
            end_pos = tuple(self.mining_target['pos'].astype(int))
            pygame.draw.aaline(self.screen, self.COLOR_BEAM, start_pos, end_pos, 2)

        # Draw asteroids
        for asteroid in self.asteroids:
            points = []
            for i in range(len(asteroid['shape_points'])):
                angle = asteroid['angle'] + (2 * math.pi * i / len(asteroid['shape_points']))
                dist = asteroid['shape_points'][i]
                x = asteroid['pos'][0] + dist * math.cos(angle)
                y = asteroid['pos'][1] + dist * math.sin(angle)
                points.append((int(x), int(y)))
            
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)

        # Draw player
        player_points = [
            (self.PLAYER_RADIUS, 0),
            (-self.PLAYER_RADIUS * 0.7, self.PLAYER_RADIUS * 0.8),
            (-self.PLAYER_RADIUS * 0.4, 0),
            (-self.PLAYER_RADIUS * 0.7, -self.PLAYER_RADIUS * 0.8),
        ]
        
        rotated_points = []
        for x, y in player_points:
            rx = x * math.cos(self.player_angle) - y * math.sin(self.player_angle)
            ry = x * math.sin(self.player_angle) + y * math.cos(self.player_angle)
            rotated_points.append((int(self.player_pos[0] + rx), int(self.player_pos[1] + ry)))
            
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"ORE: {int(self.score)} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Timer
        time_left_sec = max(0, self.time_remaining // self.fps)
        time_color = self.COLOR_TIMER_OK
        if time_left_sec < self.time_limit_seconds * 0.2:
            time_color = self.COLOR_TIMER_CRIT
        elif time_left_sec < self.time_limit_seconds * 0.5:
            time_color = self.COLOR_TIMER_WARN
            
        timer_surf = self.font_ui.render(f"TIME: {time_left_sec}", True, time_color)
        self.screen.blit(timer_surf, (self.width - timer_surf.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg_text = "MISSION COMPLETE" if self.game_won else "MISSION FAILED"
            msg_color = self.COLOR_TIMER_OK if self.game_won else self.COLOR_TIMER_CRIT
            msg_surf = self.font_msg.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(msg_surf, msg_rect)

    # --- Helper Functions for Reset/State Management ---
    def _create_stars(self):
        stars = []
        for _ in range(100):
            stars.append({
                'pos': (self.np_random.integers(0, self.width), self.np_random.integers(0, self.height)),
                'size': self.np_random.integers(1, 3),
                'color': (self.np_random.integers(50, 150), self.np_random.integers(50, 150), self.np_random.integers(100, 200))
            })
        return stars

    def _create_asteroids(self):
        asteroids = []
        total_ore_available = 0
        while total_ore_available < self.WIN_SCORE * 1.5: # Ensure enough ore exists
            asteroids.clear()
            total_ore_available = 0
            for _ in range(self.INITIAL_ASTEROIDS):
                size_choice = self.np_random.choice([15, 25, 40], p=[0.5, 0.3, 0.2])
                ore_map = {15: 3, 25: 7, 40: 15}
                ore = ore_map[size_choice]
                
                # Ensure asteroids don't spawn on player
                while True:
                    pos = self.np_random.uniform(low=0, high=[self.width, self.height])
                    if np.linalg.norm(pos - self.player_pos) > 100:
                        break
                
                shape_points = []
                for _ in range(12): # Number of vertices
                    shape_points.append(size_choice + self.np_random.uniform(-size_choice*0.3, size_choice*0.3))

                asteroids.append({
                    'pos': pos,
                    'radius': size_choice,
                    'base_radius': size_choice, # for reward
                    'ore': ore,
                    'angle': self.np_random.uniform(0, 2 * math.pi),
                    'rot_speed': self.np_random.uniform(-0.01, 0.01),
                    'shape_points': shape_points
                })
                total_ore_available += ore
        return asteroids

    def _get_dist_to_nearest_asteroid(self):
        if not self.asteroids:
            return None
        return min(np.linalg.norm(self.player_pos - a['pos']) for a in self.asteroids)

    # --- Particle Creation Helpers ---
    def _create_thrust_particles(self):
        if self.np_random.random() < 0.8:
            angle = self.player_angle + math.pi + self.np_random.uniform(-0.3, 0.3)
            speed = np.linalg.norm(self.player_vel) + 2
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            
            start_offset = np.array([math.cos(self.player_angle), math.sin(self.player_angle)]) * -self.PLAYER_RADIUS
            
            self.particles.append({
                'pos': self.player_pos + start_offset,
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'max_life': 20,
                'size': self.np_random.integers(2, 5),
                'color': self.COLOR_THRUST
            })

    def _create_ore_particles(self, asteroid):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 3)
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed
        self.particles.append({
            'pos': asteroid['pos'].copy(),
            'vel': vel,
            'life': self.np_random.integers(20, 40),
            'max_life': 40,
            'size': self.np_random.integers(1, 3),
            'color': self.COLOR_ORE
        })

    def close(self):
        pygame.quit()
        super().close()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Use a separate display for human play
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption(env.game_description)
    
    total_reward = 0
    
    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

    print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
    
    # Wait for a bit before closing
    pygame.time.wait(3000)
    env.close()