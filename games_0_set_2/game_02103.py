
# Generated: 2025-08-28T03:44:01.957086
# Source Brief: brief_02103.md
# Brief Index: 2103

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move. Press space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A modern, visually vibrant take on Space Invaders with procedurally generated alien waves and a risk/reward scoring system."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 2000
    TOTAL_WAVES = 5
    PLAYER_LIVES = 3
    PLAYER_SPEED = 6
    PLAYER_FIRE_COOLDOWN_MAX = 8  # 4 shots per second approx
    PLAYER_BULLET_SPEED = 10
    ENEMY_BULLET_SPEED = 4
    RISKY_SHOT_RADIUS = 75

    # --- Colors (Neon Aesthetic) ---
    COLOR_BG = (10, 5, 20)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_THRUSTER = (255, 100, 0)
    COLOR_PLAYER_BULLET = (255, 255, 255)
    COLOR_ENEMY_BULLET = (255, 50, 50)
    COLOR_UI = (200, 200, 255)
    ENEMY_COLORS = [
        (255, 0, 100),  # Wave 1
        (255, 80, 0),   # Wave 2
        (200, 0, 255),  # Wave 3
        (0, 255, 150),  # Wave 4
        (255, 255, 0),  # Wave 5
    ]
    PARTICLE_COLORS = [(255, 180, 0), (255, 80, 0), (200, 0, 0)]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_wave = pygame.font.Font(None, 36)
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 1
        self.player_pos = pygame.Vector2(0, 0)
        self.player_lives = 0
        self.player_fire_cooldown = 0
        self.player_bullets = []
        self.enemies = []
        self.enemy_bullets = []
        self.particles = []
        self.stars = []
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 1
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        self.player_lives = self.PLAYER_LIVES
        self.player_fire_cooldown = 0
        
        self.player_bullets = []
        self.enemy_bullets = []
        self.particles = []
        
        self._generate_stars()
        self._spawn_wave()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30) # Maintain 30 FPS

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01  # Small penalty for each step to encourage speed
        
        # Unpack factorized action
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Player movement
        player_moved = False
        old_player_pos = self.player_pos.copy()
        if movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
            player_moved = True
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
            player_moved = True
        self.player_pos.x = np.clip(self.player_pos.x, 20, self.WIDTH - 20)
        
        if player_moved:
            # Add thruster particles
            for _ in range(2):
                self._add_particle(
                    pos=self.player_pos + pygame.Vector2(0, 10),
                    vel=pygame.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(2, 4)),
                    radius=self.np_random.uniform(2, 4),
                    color=self.COLOR_PLAYER_THRUSTER,
                    lifespan=10
                )
            
            # Reward for moving towards the closest enemy
            closest_enemy = self._get_closest_enemy(self.player_pos)
            if closest_enemy:
                old_dist = old_player_pos.distance_to(closest_enemy['pos'])
                new_dist = self.player_pos.distance_to(closest_enemy['pos'])
                if new_dist < old_dist:
                    reward += 0.1

        # Player firing
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
            
        if space_held and self.player_fire_cooldown == 0:
            # sfx: player_shoot.wav
            self.player_bullets.append(pygame.Vector2(self.player_pos.x, self.player_pos.y - 15))
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN_MAX
            
            # Risky/safe shot reward
            is_risky = False
            for bullet in self.enemy_bullets:
                if self.player_pos.distance_to(bullet) < self.RISKY_SHOT_RADIUS:
                    is_risky = True
                    break
            reward += 2.0 if is_risky else -0.2

        # --- Update Game State ---
        self._update_bullets()
        self._update_enemies()
        self._update_particles()
        
        # --- Handle Collisions ---
        reward += self._handle_collisions()

        # --- Check for Wave Completion ---
        if not self.enemies:
            reward += 100
            self.wave += 1
            if self.wave > self.TOTAL_WAVES:
                # Game won
                reward += 500
                self.game_over = True
            else:
                # sfx: wave_clear.wav
                self._spawn_wave()
                self.player_bullets.clear() # Clear screen of old bullets
                self.enemy_bullets.clear()

        # --- Check Termination Conditions ---
        self.steps += 1
        terminated = self.game_over or self.player_lives <= 0 or self.steps >= self.MAX_STEPS
        if self.player_lives <= 0 and not self.game_over:
             self.game_over = True # Ensure game over state is set on final frame

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "lives": self.player_lives,
        }

    # --- Helper methods for game logic ---

    def _spawn_wave(self):
        self.enemies.clear()
        num_cols = 8
        num_rows = 3
        spacing_x = 50
        spacing_y = 40
        start_x = (self.WIDTH - (num_cols - 1) * spacing_x) / 2
        start_y = 60
        
        base_speed = 1.0 + (self.wave - 1) * 0.2
        base_fire_prob = 0.002 + (self.wave - 1) * 0.0005
        
        enemy_type = 'zigzag' if self.wave >= 3 else 'standard'
        
        for row in range(num_rows):
            for col in range(num_cols):
                pos = pygame.Vector2(start_x + col * spacing_x, start_y + row * spacing_y)
                self.enemies.append({
                    'pos': pos,
                    'initial_pos': pos.copy(),
                    'type': enemy_type,
                    'speed': base_speed,
                    'fire_prob': base_fire_prob,
                    'move_dir': 1,
                    'move_phase': self.np_random.uniform(0, 2 * math.pi) # For sine wave movement
                })
    
    def _update_enemies(self):
        wave_drift = math.sin(self.steps * 0.02) * 0.5
        
        for enemy in self.enemies:
            if enemy['type'] == 'standard':
                enemy['pos'].x = enemy['initial_pos'].x + math.sin(self.steps * 0.05 + enemy['move_phase']) * 20
                enemy['pos'].y += enemy['speed'] * 0.2
            elif enemy['type'] == 'zigzag':
                enemy['pos'].x += enemy['move_dir'] * enemy['speed']
                if enemy['pos'].x > self.WIDTH - 20 or enemy['pos'].x < 20:
                    enemy['move_dir'] *= -1
                enemy['pos'].y += enemy['speed'] * 0.15

            enemy['pos'].x += wave_drift

            # Enemy firing
            if self.np_random.random() < enemy['fire_prob']:
                # sfx: enemy_shoot.wav
                self.enemy_bullets.append(enemy['pos'].copy())

            # Check if enemy reached bottom
            if enemy['pos'].y > self.HEIGHT - 20:
                self.enemies.remove(enemy)
                self._player_hit(-10) # Penalty for letting an enemy pass

    def _update_bullets(self):
        self.player_bullets = [b for b in self.player_bullets if b.y > 0]
        for bullet in self.player_bullets:
            bullet.y -= self.PLAYER_BULLET_SPEED
            
        self.enemy_bullets = [b for b in self.enemy_bullets if b.y < self.HEIGHT]
        for bullet in self.enemy_bullets:
            bullet.y += self.ENEMY_BULLET_SPEED
            
    def _handle_collisions(self):
        reward = 0
        
        # Player bullets vs enemies
        for bullet in self.player_bullets[:]:
            for enemy in self.enemies[:]:
                if bullet.distance_to(enemy['pos']) < 15:
                    # sfx: enemy_explosion.wav
                    self._create_explosion(enemy['pos'], self.ENEMY_COLORS[self.wave - 1])
                    self.enemies.remove(enemy)
                    if bullet in self.player_bullets: self.player_bullets.remove(bullet)
                    reward += 1.0
                    self.score += 100
                    break
                    
        # Enemy bullets vs player
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 7, 20, 14)
        for bullet in self.enemy_bullets[:]:
            if player_rect.collidepoint(bullet.x, bullet.y):
                self.enemy_bullets.remove(bullet)
                reward += self._player_hit(-10)
                break # Only one hit per frame
                
        return reward

    def _player_hit(self, reward_penalty):
        self.player_lives -= 1
        self.score = max(0, self.score - 50)
        # sfx: player_explosion.wav
        self._create_explosion(self.player_pos, self.COLOR_PLAYER)
        if self.player_lives > 0:
            # Respawn player in center
            self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        return reward_penalty
        
    def _get_closest_enemy(self, pos):
        closest_enemy = None
        min_dist = float('inf')
        if not self.enemies:
            return None
        for enemy in self.enemies:
            dist = pos.distance_to(enemy['pos'])
            if dist < min_dist:
                min_dist = dist
                closest_enemy = enemy
        return closest_enemy

    # --- Particle and Starfield Methods ---

    def _add_particle(self, pos, vel, radius, color, lifespan):
        self.particles.append({
            'pos': pos.copy(), 'vel': vel.copy(), 'radius': radius, 
            'color': color, 'lifespan': lifespan, 'max_life': lifespan
        })

    def _create_explosion(self, pos, base_color):
        num_particles = 30
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            radius = self.np_random.uniform(2, 6)
            lifespan = self.np_random.integers(15, 30)
            color = self.np_random.choice(self.PARTICLE_COLORS)
            self._add_particle(pos, vel, radius, color, lifespan)

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] *= 0.95
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                'size': self.np_random.uniform(0.5, 1.5),
                'brightness': self.np_random.uniform(50, 150)
            })

    # --- Rendering methods ---

    def _render_game(self):
        self._render_stars()
        self._render_particles()
        self._render_bullets()
        if self.player_lives > 0:
            self._render_player()
        self._render_enemies()

    def _render_stars(self):
        for star in self.stars:
            brightness = star['brightness']
            if self.np_random.random() < 0.01: # Twinkle
                brightness = self.np_random.uniform(100, 200)
            color = (brightness, brightness, brightness)
            pygame.draw.circle(self.screen, color, (int(star['pos'].x), int(star['pos'].y)), star['size'])

    def _render_player(self):
        p = self.player_pos
        points = [
            (p.x, p.y - 12),
            (p.x - 12, p.y + 10),
            (p.x + 12, p.y + 10)
        ]
        pygame.gfxdraw.aapolygon(self.screen, [(int(x), int(y)) for x,y in points], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x,y in points], self.COLOR_PLAYER)
        # Cockpit
        pygame.gfxdraw.aacircle(self.screen, int(p.x), int(p.y+2), 3, (200, 255, 255))
        pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y+2), 3, (200, 255, 255))

    def _render_enemies(self):
        color = self.ENEMY_COLORS[min(self.wave - 1, len(self.ENEMY_COLORS) - 1)]
        for enemy in self.enemies:
            p = enemy['pos']
            size = 12
            points = [
                (p.x - size, p.y - size*0.5), (p.x + size, p.y - size*0.5),
                (p.x + size*0.7, p.y + size*0.7), (p.x - size*0.7, p.y + size*0.7)
            ]
            pygame.gfxdraw.aapolygon(self.screen, [(int(x), int(y)) for x,y in points], color)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x,y in points], color)
            pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y-2), 3, (255, 255, 255))
            
    def _render_bullets(self):
        for b in self.player_bullets:
            pygame.draw.line(self.screen, self.COLOR_PLAYER_BULLET, (int(b.x), int(b.y)), (int(b.x), int(b.y) + 8), 2)
        for b in self.enemy_bullets:
            pygame.gfxdraw.aacircle(self.screen, int(b.x), int(b.y), 4, self.COLOR_ENEMY_BULLET)
            pygame.gfxdraw.filled_circle(self.screen, int(b.x), int(b.y), 4, self.COLOR_ENEMY_BULLET)
            
    def _render_particles(self):
        for p in self.particles:
            alpha = 255 * (p['lifespan'] / p['max_life'])
            color = (*p['color'], alpha)
            # Pygame doesn't handle alpha in filled shapes well on surfaces without SRCALPHA flag.
            # We can simulate it by blending.
            temp_surf = pygame.Surface((int(p['radius']*2), int(p['radius']*2)), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        # Wave
        wave_str = f"WAVE {self.wave}" if self.wave <= self.TOTAL_WAVES else "VICTORY!"
        wave_text = self.font_wave.render(wave_str, True, self.COLOR_UI)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        
        # Lives
        for i in range(self.player_lives):
            p_x = self.WIDTH / 2 - (self.player_lives - 1) * 15 + i * 30
            points = [
                (p_x, self.HEIGHT - 15),
                (p_x - 8, self.HEIGHT - 5),
                (p_x + 8, self.HEIGHT - 5)
            ]
            pygame.gfxdraw.aapolygon(self.screen, [(int(x), int(y)) for x,y in points], self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x,y in points], self.COLOR_PLAYER)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    # Set a dummy video driver to run pygame headlessly
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # This block is for testing the environment's API
    print("--- Running Basic API Test ---")
    obs, info = env.reset()
    print("Initial Info:", info)
    
    terminated = False
    total_reward = 0
    for i in range(200):
        action = env.action_space.sample() # Take random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i+1) % 50 == 0:
            print(f"Step {i+1}: Info={info}, Reward={reward:.2f}")
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps.")
            print(f"Final Info: {info}, Total Reward: {total_reward:.2f}")
            break
            
    env.close()
    
    # To visualize the game (requires a display)
    # This part will not run with the dummy video driver, but shows how to use it
    # if you remove the os.environ line.
    print("\n--- To visualize, remove 'os.environ' line and run this block ---")
    # env_vis = GameEnv(render_mode="rgb_array")
    # obs, _ = env_vis.reset()
    # screen = pygame.display.set_mode((env_vis.WIDTH, env_vis.HEIGHT))
    # pygame.display.set_caption("Neon Invaders")
    #
    # running = True
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #
    #     keys = pygame.key.get_pressed()
    #     movement = 0
    #     if keys[pygame.K_LEFT]: movement = 3
    #     elif keys[pygame.K_RIGHT]: movement = 4
    #
    #     space = 1 if keys[pygame.K_SPACE] else 0
    #     shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
    #
    #     action = [movement, space, shift]
    #     obs, reward, terminated, truncated, info = env_vis.step(action)
    #
    #     # Transpose the observation back for pygame display
    #     frame = np.transpose(obs, (1, 0, 2))
    #     surf = pygame.surfarray.make_surface(frame)
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #
    #     if terminated or truncated:
    #         print("Game Over! Resetting...")
    #         env_vis.reset()
    #
    # env_vis.close()