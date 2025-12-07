
# Generated: 2025-08-28T01:46:49.072285
# Source Brief: brief_04228.md
# Brief Index: 4228

        
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
        "Controls: Arrow keys to aim. Hold Space for a short jump, or Shift for a long jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a treacherous asteroid field. Jump between safe zones, collect valuable coins, and try to reach the target score before you run out of lives."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500
        self.WIN_SCORE = 100
        self.MAX_LIVES = 3
        
        # Player constants
        self.PLAYER_SIZE = 12
        self.SHORT_JUMP_DIST = 60
        self.LONG_JUMP_DIST = 120

        # Entity counts
        self.NUM_ASTEROIDS = 8
        self.NUM_COINS = 5
        
        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_OUTLINE = (200, 220, 255)
        self.COLOR_ASTEROID = (120, 80, 80)
        self.COLOR_ASTEROID_OUTLINE = (180, 150, 150)
        self.COLOR_COIN = (255, 220, 0)
        self.COLOR_BONUS_COIN = (0, 255, 150)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE_EXPLOSION = [(255, 50, 50), (255, 150, 50)]
        self.COLOR_PARTICLE_COLLECT = [(255, 255, 255), self.COLOR_COIN]
        self.COLOR_PARTICLE_JUMP = [self.COLOR_PLAYER, (150, 200, 255)]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # Initialize state variables
        self.player_pos = None
        self.player_angle = None
        self.last_move_direction = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.asteroid_speed = None
        self.asteroids = []
        self.coins = []
        self.particles = []
        self.stars = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.player_angle = 0.0
        self.last_move_direction = np.array([1.0, 0.0]) # Start facing right
        self.lives = self.MAX_LIVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.asteroid_speed = 1.5
        
        self.particles.clear()
        self._spawn_initial_entities()
        
        return self._get_observation(), self._get_info()

    def _spawn_initial_entities(self):
        self.asteroids.clear()
        self.coins.clear()
        self.stars.clear()

        # Generate a static starfield
        for _ in range(150):
            self.stars.append(
                (
                    self.np_random.integers(0, self.WIDTH),
                    self.np_random.integers(0, self.HEIGHT),
                    self.np_random.uniform(0.5, 1.5) # star size
                )
            )

        # Spawn asteroids
        for _ in range(self.NUM_ASTEROIDS):
            self._spawn_asteroid(initial_spawn=True)

        # Spawn coins
        for _ in range(self.NUM_COINS):
            self._spawn_coin()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # 1. Handle Input & Player Movement
        jump_reward = self._handle_input(action)
        reward += jump_reward
        
        # 2. Update Game State
        self._update_asteroids()
        self._update_particles()
        
        # 3. Handle Collisions & Events
        event_reward = self._handle_collisions()
        reward += event_reward
        
        # 4. Update counters & difficulty
        self.steps += 1
        if self.steps > 0 and self.steps % 200 == 0:
            self.asteroid_speed = min(self.asteroid_speed + 0.05, 4.0)
        
        # 5. Check Termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100
            elif self.lives <= 0:
                reward -= 50
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Update aim direction
        if movement == 1: # Up
            self.last_move_direction = np.array([0, -1.0])
            self.player_angle = -math.pi / 2
        elif movement == 2: # Down
            self.last_move_direction = np.array([0, 1.0])
            self.player_angle = math.pi / 2
        elif movement == 3: # Left
            self.last_move_direction = np.array([-1.0, 0])
            self.player_angle = math.pi
        elif movement == 4: # Right
            self.last_move_direction = np.array([1.0, 0])
            self.player_angle = 0
        
        # Handle jump
        jump_dist = 0
        if shift_held:
            jump_dist = self.LONG_JUMP_DIST
        elif space_held:
            jump_dist = self.SHORT_JUMP_DIST

        if jump_dist > 0:
            # Continuous reward for jumping towards/away from coins
            dist_before = self._get_closest_coin_dist()
            
            # Perform jump
            old_pos = self.player_pos.copy()
            self.player_pos += self.last_move_direction * jump_dist
            self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
            self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)
            
            # Sound placeholder
            # if shift_held: play_long_jump_sound() else: play_short_jump_sound()

            dist_after = self._get_closest_coin_dist()
            if dist_after < dist_before:
                reward += 0.1 # Moved closer
            else:
                reward -= 0.01 # Moved further or stayed same

            # Create jump trail particles
            for i in range(10):
                p_pos = old_pos + (self.player_pos - old_pos) * (i / 10.0)
                self._create_particles(p_pos, 1, self.COLOR_PARTICLE_JUMP, 5, 0.5)

        return reward

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'][1] += self.asteroid_speed
            if asteroid['pos'][1] - asteroid['radius'] > self.HEIGHT:
                self._spawn_asteroid(existing_asteroid=asteroid)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.98

    def _handle_collisions(self):
        reward = 0
        
        # Player vs Asteroids
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_SIZE + asteroid['radius']:
                self.lives -= 1
                reward -= 10
                self._create_particles(self.player_pos, 30, self.COLOR_PARTICLE_EXPLOSION, 20, 3)
                self._spawn_asteroid(existing_asteroid=asteroid) # Respawn collided asteroid
                # Sound placeholder: play_explosion_sound()
                if self.lives > 0:
                    # Brief invulnerability and reset to center
                    self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
                break # Only one collision per frame

        # Player vs Coins
        for coin in self.coins[:]:
            dist = np.linalg.norm(self.player_pos - coin['pos'])
            if dist < self.PLAYER_SIZE + coin['radius']:
                if coin['type'] == 'bonus':
                    self.score += 5
                    reward += 5
                else:
                    self.score += 1
                    reward += 1
                
                self._create_particles(coin['pos'], 15, self.COLOR_PARTICLE_COLLECT, 15, 2)
                self.coins.remove(coin)
                self._spawn_coin()
                # Sound placeholder: play_coin_collect_sound()

        return reward

    def _check_termination(self):
        return self.lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _spawn_asteroid(self, initial_spawn=False, existing_asteroid=None):
        radius = self.np_random.integers(15, 35)
        pos = np.array([
            self.np_random.uniform(radius, self.WIDTH - radius),
            self.np_random.uniform(-self.HEIGHT, -radius) if not initial_spawn else self.np_random.uniform(0, self.HEIGHT)
        ])
        
        # Ensure it doesn't spawn on the player
        while np.linalg.norm(pos - self.player_pos) < radius + self.PLAYER_SIZE + 50:
             pos[0] = self.np_random.uniform(radius, self.WIDTH - radius)

        shape = []
        num_points = self.np_random.integers(7, 12)
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dist_variation = self.np_random.uniform(0.7, 1.1)
            shape.append(
                (
                    radius * dist_variation * math.cos(angle),
                    radius * dist_variation * math.sin(angle)
                )
            )

        if existing_asteroid:
            existing_asteroid.update({'pos': pos, 'radius': radius, 'shape': shape})
        else:
            self.asteroids.append({'pos': pos, 'radius': radius, 'shape': shape})

    def _spawn_coin(self):
        is_bonus = self.np_random.random() < 0.15
        radius = 12 if is_bonus else 8
        
        if is_bonus and self.asteroids:
            # Spawn bonus coin near an asteroid for risk/reward
            asteroid = self.np_random.choice(self.asteroids)
            angle = self.np_random.uniform(0, 2 * math.pi)
            dist_from_asteroid = asteroid['radius'] + radius + self.np_random.uniform(10, 30)
            pos = asteroid['pos'] + np.array([math.cos(angle), math.sin(angle)]) * dist_from_asteroid
        else:
            # Spawn regular coin randomly
            pos = np.array([
                self.np_random.uniform(radius, self.WIDTH - radius),
                self.np_random.uniform(radius, self.HEIGHT - radius)
            ])

        pos[0] = np.clip(pos[0], radius, self.WIDTH - radius)
        pos[1] = np.clip(pos[1], radius, self.HEIGHT - radius)

        self.coins.append({'pos': pos, 'radius': radius, 'type': 'bonus' if is_bonus else 'regular'})

    def _get_closest_coin_dist(self):
        if not self.coins:
            return float('inf')
        return min(np.linalg.norm(self.player_pos - coin['pos']) for coin in self.coins)

    def _create_particles(self, pos, count, colors, life, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(life // 2, life),
                'radius': self.np_random.uniform(2, 5),
                'color': random.choice(colors)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw stars
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, (200, 200, 220), (x, y), size)

        # Draw coins
        for coin in self.coins:
            pos = coin['pos'].astype(int)
            radius = int(coin['radius'])
            if coin['type'] == 'bonus':
                color = self.COLOR_BONUS_COIN
                # Pulsing glow effect
                glow_radius = radius + 4 + 3 * math.sin(self.steps * 0.2)
                glow_color = (*color, 60) # RGBA
                s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
                self.screen.blit(s, (pos[0]-glow_radius, pos[1]-glow_radius))
            else:
                color = self.COLOR_COIN
            
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (255, 255, 255))

        # Draw asteroids
        for asteroid in self.asteroids:
            points = [(p[0] + asteroid['pos'][0], p[1] + asteroid['pos'][1]) for p in asteroid['shape']]
            if len(points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID_OUTLINE)

        # Draw particles
        for p in self.particles:
            pos = p['pos'].astype(int)
            if p['radius'] > 0:
                pygame.draw.circle(self.screen, p['color'], pos, int(p['radius']))

        # Draw player
        if self.lives > 0:
            player_points = [
                (self.PLAYER_SIZE, 0),
                (-self.PLAYER_SIZE/2, -self.PLAYER_SIZE * 0.8),
                (-self.PLAYER_SIZE/2, self.PLAYER_SIZE * 0.8)
            ]
            
            # Rotate points
            cos_a, sin_a = math.cos(self.player_angle), math.sin(self.player_angle)
            rotated_points = [
                (p[0] * cos_a - p[1] * sin_a, p[0] * sin_a + p[1] * cos_a)
                for p in player_points
            ]
            
            # Translate points to player position
            screen_points = [
                (p[0] + self.player_pos[0], p[1] + self.player_pos[1])
                for p in rotated_points
            ]
            
            pygame.gfxdraw.filled_polygon(self.screen, screen_points, self.COLOR_PLAYER)
            pygame.gfxdraw.aapolygon(self.screen, screen_points, self.COLOR_PLAYER_OUTLINE)
    
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        life_icon_points = [(10, 0), (-5, -8), (-5, 8)]
        for i in range(self.lives):
            pos_x = self.WIDTH - 25 - (i * 25)
            screen_points = [(p[0] + pos_x, p[1] + 20) for p in life_icon_points]
            pygame.gfxdraw.filled_polygon(self.screen, screen_points, self.COLOR_PLAYER)
            pygame.gfxdraw.aapolygon(self.screen, screen_points, self.COLOR_PLAYER_OUTLINE)

        # Game Over message
        if self.game_over:
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = self.COLOR_BONUS_COIN if self.score >= self.WIN_SCORE else self.COLOR_PARTICLE_EXPLOSION[0]
            
            # Text with shadow
            text_surf = self.font_game_over.render(msg, True, color)
            shadow_surf = self.font_game_over.render(msg, True, (0,0,0))
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(shadow_surf, text_rect.move(3,3))
            self.screen.blit(text_surf, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Jumper")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)

    while running:
        # --- Action mapping for human play ---
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
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Score: {info['score']}, Total Reward: {total_reward:.2f}. Press 'R' to restart.")
            # Wait for reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        print("Resetting game...")
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False
                clock.tick(env.FPS)


        clock.tick(env.FPS)
        
    env.close()