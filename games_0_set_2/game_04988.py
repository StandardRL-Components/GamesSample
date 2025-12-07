
# Generated: 2025-08-28T03:38:37.099338
# Source Brief: brief_04988.md
# Brief Index: 4988

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move. Hold Space to fire. Good luck, pilot."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A visually stunning top-down space shooter. Survive five waves of increasingly aggressive alien invaders."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.NUM_WAVES = 5
        self.INITIAL_LIVES = 3
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_PLAYER_PROJECTILE = (100, 255, 255)
        self.COLOR_ENEMY_PROJECTILE = (255, 50, 50)
        self.ALIEN_COLORS = [
            (200, 100, 255), # Wave 1: Purple
            (255, 150, 50),  # Wave 2: Orange
            (255, 255, 100), # Wave 3: Yellow
            (100, 150, 255), # Wave 4: Blue
            (255, 100, 100), # Wave 5: Red
        ]
        self.COLOR_WHITE = (255, 255, 255)

        # Game constants
        self.PLAYER_SPEED = 6.0
        self.PLAYER_PROJECTILE_SPEED = 10.0
        self.ENEMY_PROJECTILE_SPEED = 4.0
        self.PLAYER_FIRE_COOLDOWN_MAX = 6 # frames
        self.PLAYER_HIT_INVULNERABILITY = 90 # frames
        self.WAVE_TRANSITION_FRAMES = self.FPS * 3

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_pos = [0, 0]
        self.player_lives = 0
        self.player_is_hit = False
        self.player_hit_timer = 0
        self.player_fire_cooldown = 0
        self.current_wave = 0
        self.wave_transition_timer = 0
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.aliens = []
        self.particles = []
        self.stars = []
        self.np_random = None
        
        # Initialize state variables
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Player state
        self.player_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50]
        self.player_lives = self.INITIAL_LIVES
        self.player_is_hit = False
        self.player_hit_timer = 0
        self.player_fire_cooldown = 0
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        # Wave management
        self.current_wave = -1 # Will be incremented to 0
        self.wave_transition_timer = self.WAVE_TRANSITION_FRAMES // 2
        
        # Entity lists
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.aliens = []
        self.particles = []
        
        self._create_stars()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        reward = 0.0

        # Handle wave transitions
        if self.wave_transition_timer > 0:
            self.wave_transition_timer -= 1
            if self.wave_transition_timer == 0:
                self._setup_next_wave()
            return self._get_observation(), 0, False, False, self._get_info()

        # --- Regular game loop ---
        self.steps += 1
        reward += 0.1  # Survival reward

        # Update player state
        reward += self._update_player(action)
        
        # Update game objects
        self._update_projectiles()
        self._update_aliens()
        self._update_particles()
        
        # Handle collisions and calculate rewards
        reward += self._handle_collisions()
        
        # Check for wave completion
        if not self.aliens and not self.game_over:
            reward += 100  # Wave clear bonus
            if self.current_wave >= self.NUM_WAVES - 1:
                self.win = True
                self.game_over = True
                reward += 500  # Win bonus
            else:
                self.wave_transition_timer = self.WAVE_TRANSITION_FRAMES
        
        # Check termination conditions
        if self.player_lives <= 0 and not self.game_over:
            self.game_over = True
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

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
            "wave": self.current_wave + 1,
            "lives": self.player_lives,
        }

    def _update_player(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if self.player_is_hit:
            self.player_hit_timer -= 1
            if self.player_hit_timer <= 0:
                self.player_is_hit = False

        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1

        if not self.player_is_hit:
            if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
            if movement == 2: self.player_pos[1] += self.PLAYER_SPEED
            if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
            if movement == 4: self.player_pos[0] += self.PLAYER_SPEED

        self.player_pos[0] = np.clip(self.player_pos[0], 15, self.SCREEN_WIDTH - 15)
        self.player_pos[1] = np.clip(self.player_pos[1], 15, self.SCREEN_HEIGHT - 15)

        fire_reward = 0.0
        if space_held and self.player_fire_cooldown == 0 and not self.player_is_hit:
            # # Play laser sound
            self.player_projectiles.append({
                "pos": [self.player_pos[0], self.player_pos[1] - 15],
            })
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN_MAX
            fire_reward = -0.2
        
        return fire_reward

    def _update_projectiles(self):
        for p in self.player_projectiles[:]:
            p['pos'][1] -= self.PLAYER_PROJECTILE_SPEED
            if p['pos'][1] < 0:
                self.player_projectiles.remove(p)

        for p in self.enemy_projectiles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            if not (0 < p['pos'][0] < self.SCREEN_WIDTH and 0 < p['pos'][1] < self.SCREEN_HEIGHT):
                self.enemy_projectiles.remove(p)

    def _update_aliens(self):
        for alien in self.aliens:
            self._move_alien(alien)
            fire_rate = 0.01 + self.current_wave * 0.01
            if self.np_random.random() < fire_rate:
                self._alien_fire(alien)

    def _move_alien(self, alien):
        state = alien['pattern_state']
        state['t'] = (state.get('t', 0) + 1) % 10000
        t = state['t']
        speed = 1.0 + self.current_wave * 0.2
        
        wave_patterns = [
            lambda: (state['base_x'] + math.sin(t * state['freq']) * state['amp'], state['base_y']),
            lambda: (alien['pos'][0] + state['vx'], alien['pos'][1] + state['vy']),
            lambda: (state['cx'] + math.cos(t * state['freq']) * state['r'], state['cy'] + math.sin(t * state['freq']) * state['r']),
            lambda: (alien['pos'][0] + (self.player_pos[0] - alien['pos'][0]) * 0.01 * speed, alien['pos'][1] + speed * 0.5),
            lambda: (state['base_x'] + math.sin(t * state['freq']) * state['amp'], state['base_y'] + math.cos(t * state['freq'] * 0.7) * state['amp_y']),
        ]
        
        new_pos = wave_patterns[alien['type']]()
        alien['pos'][0] = new_pos[0]
        alien['pos'][1] = new_pos[1]

        if alien['type'] == 1: # Bounce off walls
            if not (20 < alien['pos'][0] < self.SCREEN_WIDTH - 20): state['vx'] *= -1

    def _alien_fire(self, alien):
        # # Play enemy fire sound
        dx = self.player_pos[0] - alien['pos'][0]
        dy = self.player_pos[1] - alien['pos'][1]
        dist = math.hypot(dx, dy)
        if dist == 0: return
        
        vel_x = (dx / dist) * self.ENEMY_PROJECTILE_SPEED
        vel_y = (dy / dist) * self.ENEMY_PROJECTILE_SPEED
        self.enemy_projectiles.append({"pos": list(alien['pos']), "vel": [vel_x, vel_y]})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['radius'] -= p['decay']
            if p['lifespan'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0.0
        
        for p in self.player_projectiles[:]:
            for a in self.aliens[:]:
                if math.hypot(p['pos'][0] - a['pos'][0], p['pos'][1] - a['pos'][1]) < 15:
                    if p in self.player_projectiles: self.player_projectiles.remove(p)
                    self.aliens.remove(a)
                    self.score += 10
                    reward += 1
                    self._create_explosion(a['pos'], 20, self.ALIEN_COLORS[a['type']])
                    # # Play alien explosion sound
                    break
        
        if not self.player_is_hit:
            for p in self.enemy_projectiles[:]:
                if math.hypot(p['pos'][0] - self.player_pos[0], p['pos'][1] - self.player_pos[1]) < 12:
                    self.enemy_projectiles.remove(p)
                    self.player_lives -= 1
                    reward -= 10
                    self.player_is_hit = True
                    self.player_hit_timer = self.PLAYER_HIT_INVULNERABILITY
                    self._create_explosion(self.player_pos, 40, self.COLOR_PLAYER)
                    # # Play player hit sound
                    if self.player_lives > 0:
                        self.player_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50]
                    break
        return reward

    def _setup_next_wave(self):
        self.current_wave += 1
        num_aliens = 8 + self.current_wave * 2
        speed = 1.0 + self.current_wave * 0.2
        
        for i in range(num_aliens):
            alien_type = self.current_wave
            
            # Pattern-specific initial state
            pattern_state = {}
            if alien_type == 0: # Horizontal sine wave
                x = 100 + i * (self.SCREEN_WIDTH - 200) / (num_aliens - 1)
                y = 80
                pattern_state = {"t": i * 10, "base_x": x, "base_y": y, "amp": 40, "freq": 0.02}
            elif alien_type == 1: # Diagonal sweep
                x = 50 + (i % 2) * (self.SCREEN_WIDTH - 100)
                y = 60 + (i // 2) * 30
                pattern_state = {"vx": speed if x < self.SCREEN_WIDTH / 2 else -speed, "vy": speed * 0.1}
            elif alien_type == 2: # Circle
                x, y = 0, 0 # Position determined by pattern
                pattern_state = {"t": i * 20, "cx": self.SCREEN_WIDTH / 2, "cy": 120, "r": 100, "freq": 0.02}
            elif alien_type == 3: # Homing swarm
                x = self.np_random.uniform(50, self.SCREEN_WIDTH - 50)
                y = self.np_random.uniform(50, 100)
            elif alien_type == 4: # Final wave complex pattern
                x = 100 + i * (self.SCREEN_WIDTH - 200) / (num_aliens - 1)
                y = 120
                pattern_state = {"t": i * 5, "base_x": x, "base_y": y, "amp": 150, "amp_y": 40, "freq": 0.015}

            self.aliens.append({
                "pos": [x, y],
                "type": alien_type,
                "pattern_state": pattern_state
            })
            self._move_alien(self.aliens[-1]) # Set initial position from pattern

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            lifespan = self.np_random.integers(self.FPS // 2, self.FPS)
            radius = self.np_random.uniform(2, 6)
            self.particles.append({
                'pos': list(pos), 'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': radius, 'lifespan': lifespan, 'max_lifespan': lifespan,
                'color': color, 'decay': radius / lifespan
            })

    def _create_stars(self):
        self.stars = []
        for _ in range(200):
            self.stars.append({
                "pos": (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)),
                "size": self.np_random.choice([1, 2]),
                "color": self.np_random.choice([(100, 100, 120), (150, 150, 170), (200, 200, 220)])
            })

    def _render_game(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star["color"], star["pos"], star["size"])
        
        for p in self.enemy_projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_ENEMY_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, self.COLOR_ENEMY_PROJECTILE)

        for p in self.player_projectiles:
            start = (int(p['pos'][0]), int(p['pos'][1]))
            end = (int(p['pos'][0]), int(p['pos'][1] + 8))
            pygame.draw.line(self.screen, self.COLOR_PLAYER_PROJECTILE, start, end, 3)
            pygame.draw.circle(self.screen, self.COLOR_WHITE, start, 2)

        for a in self.aliens: self._render_alien(a)
        if self.player_lives > 0: self._render_player()
            
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color_with_alpha = p['color'] + (alpha,)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(max(0, p['radius']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color_with_alpha)

    def _render_player(self):
        if self.player_is_hit and self.player_hit_timer % 6 < 3: return
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        points = [(x, y - 12), (x - 10, y + 8), (x + 10, y + 8)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, x, y, 3, (200, 255, 255)) # Cockpit

    def _render_alien(self, alien):
        x, y = int(alien['pos'][0]), int(alien['pos'][1])
        color = self.ALIEN_COLORS[alien['type']]
        glow_color = color + (60,)
        
        shapes = [
            lambda: pygame.gfxdraw.filled_ellipse(self.screen, x, y, 12, 6, color), # Saucer
            lambda: pygame.gfxdraw.filled_polygon(self.screen, [(x, y-8), (x-8, y+6), (x+8, y+6)], color), # Triangle
            lambda: pygame.gfxdraw.filled_polygon(self.screen, [(x, y-10), (x-7, y), (x, y+10), (x+7, y)], color), # Diamond
            lambda: pygame.gfxdraw.filled_circle(self.screen, x, y, 7, color), # Bug
            lambda: pygame.gfxdraw.filled_polygon(self.screen, [(x, y-10), (x-5, y+8), (x, y+3), (x+5, y+8)], color), # Fighter
        ]
        glow_shapes = [
            lambda: pygame.gfxdraw.filled_ellipse(self.screen, x, y, 14, 8, glow_color),
            lambda: pygame.gfxdraw.filled_polygon(self.screen, [(x, y-10), (x-10, y+8), (x+10, y+8)], glow_color),
            lambda: pygame.gfxdraw.filled_polygon(self.screen, [(x, y-12), (x-9, y), (x, y+12), (x+9, y)], glow_color),
            lambda: pygame.gfxdraw.filled_circle(self.screen, x, y, 9, glow_color),
            lambda: pygame.gfxdraw.filled_polygon(self.screen, [(x, y-12), (x-7, y+10), (x, y+5), (x+7, y+10)], glow_color),
        ]
        glow_shapes[alien['type']]()
        shapes[alien['type']]()

    def _render_ui(self):
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))

        wave_str = f"WAVE: {self.current_wave + 1}/{self.NUM_WAVES}" if self.current_wave < self.NUM_WAVES else "VICTORY"
        wave_text = self.font_medium.render(wave_str, True, self.COLOR_WHITE)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        for i in range(self.player_lives):
            x, y = 20 + i * 25, self.SCREEN_HEIGHT - 25
            points = [(x, y - 8), (x - 7, y + 5), (x + 7, y + 5)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        if self.wave_transition_timer > 0 and not self.game_over:
            alpha = int(255 * math.sin(math.pi * (1 - self.wave_transition_timer / self.WAVE_TRANSITION_FRAMES)))
            msg = f"WAVE {self.current_wave + 2}"
            text = self.font_large.render(msg, True, self.COLOR_WHITE)
            text.set_alpha(alpha)
            self.screen.blit(text, text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)))

        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_large.render(message, True, color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)))

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Wave Invaders")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    action = env.action_space.sample() # Start with a random action
    
    print(env.user_guide)
    
    while not terminated:
        # Human controls
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
    print(f"Game Over! Final Score: {info['score']}")
    env.close()