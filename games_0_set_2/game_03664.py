
# Generated: 2025-08-28T00:02:02.362554
# Source Brief: brief_03664.md
# Brief Index: 3664

        
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
        "Controls: ↑ to thrust, ←→ to turn. Hold space to activate the mining beam. Avoid asteroids!"
    )

    game_description = (
        "Pilot a mining ship through a dense asteroid field. Use your mining beam to collect valuable ore from asteroids, but be careful not to collide with them. Reach 100 ore to win, but lose all 3 lives and it's game over."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 64)
        
        # --- Game Constants ---
        self.FPS = 30
        self.WIN_SCORE = 100
        self.MAX_LIVES = 3
        self.MAX_STEPS = 5000
        self.PLAYER_RADIUS = 12
        self.PLAYER_MAX_HEALTH = 3
        self.PLAYER_THRUST = 0.25
        self.PLAYER_ROTATION_SPEED = 5.0
        self.PLAYER_DRAG = 0.98
        self.INVINCIBILITY_DURATION = 2 * self.FPS
        
        # --- Colors ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_STAR = (200, 200, 220)
        self.COLOR_PLAYER_OK = (0, 255, 128)
        self.COLOR_PLAYER_WARN = (255, 255, 0)
        self.COLOR_PLAYER_DANGER = (255, 50, 50)
        self.COLOR_ASTEROID = (120, 120, 130)
        self.COLOR_ORE = (255, 200, 0)
        self.COLOR_BEAM = (100, 200, 255, 100) # RGBA
        self.COLOR_UI_TEXT = (240, 240, 240)

        # Initialize state variables
        self.player = {}
        self.asteroids = []
        self.particles = []
        self.ore_particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.win = False
        self.difficulty_tier = 0
        self.base_asteroid_count = 5
        self.base_asteroid_speed = 0.5
        self.current_asteroid_count = 0
        self.current_asteroid_speed = 0.0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.win = False
        self.difficulty_tier = -1 # Will be set to 0 by _check_difficulty_increase

        self.player = {
            'pos': np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64),
            'vel': np.array([0.0, 0.0], dtype=np.float64),
            'angle': -90.0,
            'health': self.PLAYER_MAX_HEALTH,
            'invincible_timer': self.INVINCIBILITY_DURATION,
            'collision_cooldown': {}
        }
        
        self.asteroids.clear()
        self.particles.clear()
        self.ore_particles.clear()
        
        if not self.stars:
            self.stars = [
                (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
                for _ in range(100)
            ]
        
        self._check_difficulty_increase()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Time penalty
        
        if not self.game_over:
            # --- Handle Input & Update State ---
            self._handle_input(action)
            self._update_player()
            self._update_asteroids()
            self._update_particles()
            
            # --- Handle Interactions & Events ---
            mining_reward = self._handle_mining(action[1] == 1)
            collision_reward = self._handle_collisions()
            collection_reward = self._handle_ore_collection()
            
            # --- Calculate Rewards ---
            proximity_reward = self._calculate_proximity_reward()
            reward += mining_reward + collision_reward + collection_reward + proximity_reward

            # --- Game Progression ---
            self._check_difficulty_increase()
            if len(self.asteroids) < self.current_asteroid_count:
                self._spawn_asteroids(1)

        # --- Termination Check ---
        self.steps += 1
        terminated = self._check_termination()
        if terminated and self.win:
            reward += 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Core Logic Methods ---

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Rotation
        if movement == 3: # Left
            self.player['angle'] -= self.PLAYER_ROTATION_SPEED
        if movement == 4: # Right
            self.player['angle'] += self.PLAYER_ROTATION_SPEED
            
        # Thrust
        if movement == 1: # Up
            rad_angle = math.radians(self.player['angle'])
            thrust_vec = np.array([math.cos(rad_angle), math.sin(rad_angle)])
            self.player['vel'] += thrust_vec * self.PLAYER_THRUST
            # SFX: ship_thrust.wav
            self._create_thruster_particles()
        elif movement == 2: # Down (Brake)
            self.player['vel'] *= 0.95 # Stronger drag for braking

    def _update_player(self):
        # Apply drag and update position
        self.player['vel'] *= self.PLAYER_DRAG
        self.player['pos'] += self.player['vel']
        
        # World wrapping
        self.player['pos'][0] %= self.WIDTH
        self.player['pos'][1] %= self.HEIGHT
        
        # Update timers
        if self.player['invincible_timer'] > 0:
            self.player['invincible_timer'] -= 1
        
        cooldown_keys = list(self.player['collision_cooldown'].keys())
        for asteroid_id in cooldown_keys:
            self.player['collision_cooldown'][asteroid_id] -= 1
            if self.player['collision_cooldown'][asteroid_id] <= 0:
                del self.player['collision_cooldown'][asteroid_id]

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] += asteroid['rot_vel']
            asteroid['pos'][0] %= self.WIDTH
            asteroid['pos'][1] %= self.HEIGHT

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] - 0.1)

        self.ore_particles = [p for p in self.ore_particles if p['lifespan'] > 0]
        for p in self.ore_particles:
            # Home in on the player
            player_vec = self.player['pos'] - p['pos']
            dist = np.linalg.norm(player_vec)
            if dist > 1:
                p['vel'] = p['vel'] * 0.9 + player_vec / dist * 0.5
            p['pos'] += p['vel']
            p['lifespan'] -= 1

    def _handle_collisions(self):
        reward = 0
        asteroids_to_remove = []
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player['pos'] - asteroid['pos'])
            if dist < self.PLAYER_RADIUS + asteroid['radius'] and self.player['invincible_timer'] == 0:
                if id(asteroid) not in self.player['collision_cooldown']:
                    # SFX: collision_impact.wav
                    self.player['health'] -= 1
                    self.player['collision_cooldown'][id(asteroid)] = self.FPS // 2 # 0.5s cooldown
                    self._create_explosion(self.player['pos'], self.COLOR_PLAYER_DANGER, 10)
                    
                    if self.player['health'] <= 0:
                        # SFX: player_explosion.wav
                        self.lives -= 1
                        reward -= 10
                        self._create_explosion(self.player['pos'], self.COLOR_PLAYER_DANGER, 50, big=True)
                        if self.lives > 0:
                            self.player['pos'] = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64)
                            self.player['vel'] = np.array([0.0, 0.0], dtype=np.float64)
                            self.player['health'] = self.PLAYER_MAX_HEALTH
                            self.player['invincible_timer'] = self.INVINCIBILITY_DURATION
                        else:
                            self.game_over = True
        return reward

    def _handle_mining(self, space_held):
        reward = 0
        if not space_held:
            return 0
        
        # SFX: mining_beam_loop.wav
        beam_length = 100
        beam_angle_rad = math.radians(self.player['angle'])
        beam_end = self.player['pos'] + np.array([math.cos(beam_angle_rad), math.sin(beam_angle_rad)]) * beam_length

        for asteroid in self.asteroids:
            if asteroid['ore'] <= 0:
                continue
            
            # Simple line-circle intersection check
            ap = asteroid['pos'] - self.player['pos']
            ab = beam_end - self.player['pos']
            
            proj = np.dot(ap, ab) / np.dot(ab, ab)
            proj = np.clip(proj, 0, 1)
            
            closest_point = self.player['pos'] + proj * ab
            dist_to_beam = np.linalg.norm(asteroid['pos'] - closest_point)
            
            if dist_to_beam < asteroid['radius']:
                asteroid['ore'] -= 1
                # SFX: ore_spawn.wav
                self._create_ore_particle(asteroid['pos'])
                reward += 0.05 # Small reward for successful mining action
        return reward

    def _handle_ore_collection(self):
        reward = 0
        ore_to_remove = []
        for p in self.ore_particles:
            dist = np.linalg.norm(self.player['pos'] - p['pos'])
            if dist < self.PLAYER_RADIUS:
                # SFX: ore_collect.wav
                self.score += 1
                reward += 1
                ore_to_remove.append(p)
        
        self.ore_particles = [p for p in self.ore_particles if p not in ore_to_remove]
        return reward

    def _calculate_proximity_reward(self):
        if not self.asteroids:
            return 0
        
        closest_asteroid = min(self.asteroids, key=lambda a: np.linalg.norm(a['pos'] - self.player['pos']))
        
        vec_to_asteroid = closest_asteroid['pos'] - self.player['pos']
        dist_to_asteroid = np.linalg.norm(vec_to_asteroid)
        
        if dist_to_asteroid < 1:
            return 0

        norm_vec_to_asteroid = vec_to_asteroid / dist_to_asteroid
        
        player_vel_norm = np.linalg.norm(self.player['vel'])
        if player_vel_norm < 0.1:
            return 0

        norm_player_vel = self.player['vel'] / player_vel_norm
        
        cosine_similarity = np.dot(norm_player_vel, norm_vec_to_asteroid)
        
        # Reward for moving towards, scaled by how far away it is (more reward for closing large distances)
        return 0.1 * max(0, cosine_similarity) * (1 - 100 / (100 + dist_to_asteroid))


    # --- Spawning and Creation ---

    def _spawn_asteroids(self, num_to_spawn):
        for _ in range(num_to_spawn):
            # Spawn at edges
            edge = self.np_random.integers(4)
            if edge == 0: pos = np.array([self.np_random.uniform(0, self.WIDTH), -30.0])
            elif edge == 1: pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 30.0])
            elif edge == 2: pos = np.array([-30.0, self.np_random.uniform(0, self.HEIGHT)])
            else: pos = np.array([self.WIDTH + 30.0, self.np_random.uniform(0, self.HEIGHT)])

            vel = self.np_random.uniform(-self.current_asteroid_speed, self.current_asteroid_speed, 2)
            radius = self.np_random.uniform(15, 35)
            
            self.asteroids.append({
                'pos': pos.astype(np.float64),
                'vel': vel.astype(np.float64),
                'angle': 0,
                'rot_vel': self.np_random.uniform(-1.0, 1.0),
                'radius': radius,
                'ore': int(radius),
                'shape_points': self._create_asteroid_shape(radius)
            })

    def _create_asteroid_shape(self, radius):
        num_vertices = self.np_random.integers(7, 12)
        points = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist = self.np_random.uniform(radius * 0.8, radius * 1.2)
            points.append((dist * math.cos(angle), dist * math.sin(angle)))
        return points

    def _create_thruster_particles(self):
        rad_angle = math.radians(self.player['angle'] + 180)
        base_vel = np.array([math.cos(rad_angle), math.sin(rad_angle)]) * 2
        for _ in range(2):
            vel_spread = self.np_random.uniform(-0.5, 0.5, 2)
            self.particles.append({
                'pos': self.player['pos'].copy(),
                'vel': base_vel + vel_spread,
                'lifespan': self.np_random.integers(10, 20),
                'color': random.choice([(0, 150, 255), (100, 200, 255)]),
                'size': self.np_random.uniform(3, 6)
            })

    def _create_explosion(self, pos, color, count, big=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6) if big else self.np_random.uniform(0.5, 3)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(20, 50),
                'color': color,
                'size': self.np_random.uniform(2, 8) if big else self.np_random.uniform(1, 4)
            })
    
    def _create_ore_particle(self, pos):
        self.ore_particles.append({
            'pos': pos.copy(),
            'vel': self.np_random.uniform(-1, 1, 2).astype(np.float64),
            'lifespan': self.np_random.integers(100, 150),
        })

    # --- Difficulty Management ---
    def _check_difficulty_increase(self):
        new_tier = self.score // 25
        if new_tier > self.difficulty_tier:
            self.difficulty_tier = new_tier
            self.current_asteroid_count = int(self.base_asteroid_count * (1.1 ** self.difficulty_tier))
            self.current_asteroid_speed = self.base_asteroid_speed + 0.25 * self.difficulty_tier
            # Spawn missing asteroids on level up
            num_to_spawn = self.current_asteroid_count - len(self.asteroids)
            if num_to_spawn > 0:
                self._spawn_asteroids(num_to_spawn)

    # --- Rendering Methods ---
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render stars
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))

        # Render particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'].astype(int), int(p['size']))
        
        # Render ore particles
        for p in self.ore_particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, self.COLOR_ORE)
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, self.COLOR_ORE)
            
        # Render mining beam
        if not self.game_over and self.action_space.sample()[1] == 1: # A bit of a hack to check if space could be held
             if any(a['ore'] > 0 for a in self.asteroids):
                self._render_mining_beam()

        # Render asteroids
        for asteroid in self.asteroids:
            self._render_asteroid(asteroid)

        # Render player
        if self.lives > 0:
            self._render_player()
            
    def _render_player(self):
        pos_int = self.player['pos'].astype(int)
        rad_angle = math.radians(self.player['angle'])
        
        # Create triangle points
        p1 = pos_int + np.array([math.cos(rad_angle), math.sin(rad_angle)]) * self.PLAYER_RADIUS
        p2 = pos_int + np.array([math.cos(rad_angle + 2.5), math.sin(rad_angle + 2.5)]) * self.PLAYER_RADIUS
        p3 = pos_int + np.array([math.cos(rad_angle - 2.5), math.sin(rad_angle - 2.5)]) * self.PLAYER_RADIUS
        points = [p1.tolist(), p2.tolist(), p3.tolist()]
        
        # Health color
        if self.player['health'] == 3: color = self.COLOR_PLAYER_OK
        elif self.player['health'] == 2: color = self.COLOR_PLAYER_WARN
        else: color = self.COLOR_PLAYER_DANGER
        
        # Invincibility flash
        if self.player['invincible_timer'] > 0 and self.steps % 10 < 5:
            color = (255, 255, 255)

        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_asteroid(self, asteroid):
        rotated_points = []
        for x, y in asteroid['shape_points']:
            rad_angle = math.radians(asteroid['angle'])
            new_x = x * math.cos(rad_angle) - y * math.sin(rad_angle)
            new_y = x * math.sin(rad_angle) + y * math.cos(rad_angle)
            rotated_points.append((int(new_x + asteroid['pos'][0]), int(new_y + asteroid['pos'][1])))
        
        color = self.COLOR_ASTEROID
        if asteroid['ore'] > 0:
            # Interpolate color based on ore content
            ore_ratio = min(1, asteroid['ore'] / 35.0) # 35 is max possible ore
            color = (
                int(self.COLOR_ASTEROID[0] + (self.COLOR_ORE[0] - self.COLOR_ASTEROID[0]) * ore_ratio * 0.5),
                int(self.COLOR_ASTEROID[1] + (self.COLOR_ORE[1] - self.COLOR_ASTEROID[1]) * ore_ratio * 0.5),
                int(self.COLOR_ASTEROID[2] + (self.COLOR_ORE[2] - self.COLOR_ASTEROID[2]) * ore_ratio * 0.5)
            )

        pygame.gfxdraw.aapolygon(self.screen, rotated_points, color)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, color)

    def _render_mining_beam(self):
        beam_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        
        rad_angle = math.radians(self.player['angle'])
        beam_length = 100
        beam_width = 30
        
        p1 = self.player['pos']
        p2 = p1 + np.array([math.cos(rad_angle + 0.5 * math.pi), math.sin(rad_angle + 0.5 * math.pi)]) * (beam_width / 2)
        p3 = p1 + np.array([math.cos(rad_angle), math.sin(rad_angle)]) * beam_length
        p4 = p1 + np.array([math.cos(rad_angle - 0.5 * math.pi), math.sin(rad_angle - 0.5 * math.pi)]) * (beam_width / 2)
        
        points = [p2.tolist(), p3.tolist(), p4.tolist()]
        pygame.gfxdraw.aapolygon(beam_surf, points, self.COLOR_BEAM)
        pygame.gfxdraw.filled_polygon(beam_surf, points, self.COLOR_BEAM)
        self.screen.blit(beam_surf, (0, 0))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"ORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Game Over / Win Message
        if self.game_over:
            msg = "GAME OVER" if not self.win else "YOU WIN!"
            msg_text = self.font_msg.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = msg_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_text, text_rect)

    # --- Getters and Termination ---
    
    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.win = True
            self.game_over = True
        if self.lives <= 0:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "player_health": self.player.get('health', 0),
            "difficulty_tier": self.difficulty_tier
        }

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Requires pygame to be installed with display support
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # --- Pygame setup for interactive play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    clock = pygame.time.Clock()
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Action Mapping from Keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Reset on termination ---
        if terminated and keys[pygame.K_r]:
            obs, info = env.reset()
            terminated = False

        # --- Frame Rate ---
        clock.tick(env.FPS)
        
    pygame.quit()