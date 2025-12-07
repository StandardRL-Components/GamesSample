import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a retro top-down arcade shooter.
    The player must defend Earth from waves of descending alien invaders.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Use arrow keys to move. Press space to fire your weapon. Hold shift to activate a temporary shield."
    )
    game_description = (
        "Defend Earth from waves of descending alien invaders in this retro top-down shooter. Dodge projectiles and destroy all aliens to win."
    )

    # Frame advance setting
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (10, 10, 25)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_SHIELD = (100, 255, 255, 100)
    COLOR_PLAYER_PROJECTILE = (255, 255, 100)
    COLOR_ALIEN_RED = (255, 50, 50)
    COLOR_ALIEN_BLUE = (100, 150, 255)
    COLOR_ALIEN_PURPLE = (200, 100, 255)
    COLOR_ALIEN_PROJECTILE = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_EXPLOSION = [(255, 255, 100), (255, 150, 50), (255, 50, 50)]

    # Game dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Player settings
    PLAYER_SIZE = 12
    PLAYER_SPEED = 5
    PLAYER_FIRE_COOLDOWN = 6  # frames
    PLAYER_LIVES = 3
    SHIELD_DURATION = 5  # frames
    SHIELD_COOLDOWN = 90 # frames

    # Alien settings
    ALIEN_SIZE = 10
    ALIEN_ROWS = 3
    ALIEN_COLS = 10
    TOTAL_ALIENS = 30

    # Projectile settings
    PROJECTILE_SPEED = 10
    PROJECTILE_SIZE = 3

    # Episode settings
    MAX_STEPS = 3000 # Increased to allow for longer gameplay

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)

        self.stars = []
        self.player_pos = [0, 0]
        self.player_lives = 0
        self.player_fire_timer = 0
        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown_timer = 0
        self.player_projectiles = []
        self.aliens = []
        self.alien_projectiles = []
        self.explosions = []
        self.current_wave = 0
        self.aliens_destroyed_total = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.rng = None

        self._create_starfield()
        # self.validate_implementation() # Commented out for submission


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.player_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 30]
        self.player_lives = self.PLAYER_LIVES
        self.player_fire_timer = 0

        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown_timer = 0

        self.player_projectiles = []
        self.aliens = []
        self.alien_projectiles = []
        self.explosions = []

        self.current_wave = 0
        self.aliens_destroyed_total = 0
        self._spawn_wave()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.1  # Survival reward

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_input(movement, space_held, shift_held)
        self._update_game_state()
        step_reward = self._handle_collisions()
        reward += step_reward

        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated:
            if self.game_won:
                reward += 100
            else:
                reward -= 100
        
        if truncated and not terminated:
            reward -= 50 # Penalty for running out of time

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED # Up
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED # Down
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED # Left
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED # Right

        # Clamp player position
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.SCREEN_HEIGHT - 100, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

        # Firing
        if space_held and self.player_fire_timer <= 0:
            # sfx: player_shoot.wav
            self.player_projectiles.append(pygame.Rect(self.player_pos[0] - self.PROJECTILE_SIZE // 2, self.player_pos[1], self.PROJECTILE_SIZE, self.PROJECTILE_SIZE * 2))
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN

        # Shield
        if shift_held and not self.shield_active and self.shield_cooldown_timer <= 0:
            # sfx: shield_activate.wav
            self.shield_active = True
            self.shield_timer = self.SHIELD_DURATION
            self.shield_cooldown_timer = self.SHIELD_COOLDOWN

    def _update_game_state(self):
        # Timers
        if self.player_fire_timer > 0: self.player_fire_timer -= 1
        if self.shield_cooldown_timer > 0: self.shield_cooldown_timer -= 1
        if self.shield_timer > 0:
            self.shield_timer -= 1
            if self.shield_timer <= 0:
                self.shield_active = False

        # Player Projectiles
        self.player_projectiles = [p for p in self.player_projectiles if p.y > -p.height]
        for p in self.player_projectiles: p.y -= self.PROJECTILE_SPEED

        # Alien Projectiles
        updated_alien_projectiles = []
        for p_data in self.alien_projectiles:
            p_data['rect'].y += p_data['speed']
            if 'vel_x' in p_data:
                p_data['rect'].x += p_data['vel_x']
            
            if p_data['rect'].y < self.SCREEN_HEIGHT and p_data['rect'].right > 0 and p_data['rect'].left < self.SCREEN_WIDTH:
                updated_alien_projectiles.append(p_data)
        self.alien_projectiles = updated_alien_projectiles

        # Aliens
        alien_speed = 0.5 + self.current_wave * 0.2
        for alien in self.aliens:
            alien['pos'][1] += alien_speed
            alien['bob_offset'] = math.sin(self.steps * 0.1 + alien['pos'][0] * 0.1) * 3
            
            # Alien Firing
            alien['fire_timer'] -= 1
            if alien['fire_timer'] <= 0:
                self._alien_fire(alien)

        # Explosions
        for explosion in self.explosions:
            explosion['radius'] += explosion['speed']
            explosion['alpha'] = max(0, explosion['alpha'] - 10)
        self.explosions = [e for e in self.explosions if e['alpha'] > 0]

    def _alien_fire(self, alien):
        fire_chance = (0.005 + self.current_wave * 0.005)
        if self.rng.random() < fire_chance:
            # sfx: alien_shoot.wav
            base_cooldown = 40 - self.current_wave * 5
            alien['fire_timer'] = self.rng.integers(base_cooldown, base_cooldown + 20)
            
            start_x, start_y = alien['pos'][0], alien['pos'][1]

            if alien['type'] == 'red':
                proj_rect = pygame.Rect(start_x - self.PROJECTILE_SIZE // 2, start_y, self.PROJECTILE_SIZE, self.PROJECTILE_SIZE)
                self.alien_projectiles.append({'rect': proj_rect, 'speed': 4})
            elif alien['type'] == 'blue':
                for i in [-1, 0, 1]:
                    angle = math.pi / 2 + i * 0.3
                    vel = [math.cos(angle) * 4, math.sin(angle) * 4]
                    p_rect = pygame.Rect(start_x - self.PROJECTILE_SIZE // 2, start_y, self.PROJECTILE_SIZE, self.PROJECTILE_SIZE)
                    self.alien_projectiles.append({'rect': p_rect, 'speed': vel[1], 'vel_x': vel[0]})
            elif alien['type'] == 'purple':
                dx = self.player_pos[0] - start_x
                dy = self.player_pos[1] - start_y
                dist = math.hypot(dx, dy)
                if dist > 0:
                    vel = [dx / dist * 5, dy / dist * 5]
                    p_rect = pygame.Rect(start_x - self.PROJECTILE_SIZE // 2, start_y, self.PROJECTILE_SIZE, self.PROJECTILE_SIZE)
                    self.alien_projectiles.append({'rect': p_rect, 'speed': vel[1], 'vel_x': vel[0]})

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE // 2, self.player_pos[1] - self.PLAYER_SIZE // 2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Player projectiles vs Aliens
        for p in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                alien_rect = pygame.Rect(alien['pos'][0] - self.ALIEN_SIZE, alien['pos'][1] - self.ALIEN_SIZE, self.ALIEN_SIZE * 2, self.ALIEN_SIZE * 2)
                if alien_rect.colliderect(p):
                    # sfx: explosion.wav
                    self._create_explosion(alien['pos'])
                    self.aliens.remove(alien)
                    if p in self.player_projectiles: self.player_projectiles.remove(p)
                    self.score += 10
                    reward += 10
                    self.aliens_destroyed_total += 1
                    break
        
        # Check for wave completion
        if not self.aliens and self.aliens_destroyed_total < self.TOTAL_ALIENS:
            self._spawn_wave()

        # Alien projectiles vs Player
        for p_data in self.alien_projectiles[:]:
            if player_rect.colliderect(p_data['rect']):
                if self.shield_active:
                    # sfx: shield_block.wav
                    self.alien_projectiles.remove(p_data)
                    reward += 2 # Reward for successful block
                else:
                    # sfx: player_hit.wav
                    self.alien_projectiles.remove(p_data)
                    self.player_lives -= 1
                    reward -= 5
                    self._create_explosion(self.player_pos, is_player=True)
                break
        
        return reward

    def _check_termination(self):
        if self.player_lives <= 0:
            self.game_over = True
            return True
        if self.aliens_destroyed_total >= self.TOTAL_ALIENS:
            self.game_over = True
            self.game_won = True
            return True
        for alien in self.aliens:
            if alien['pos'][1] > self.SCREEN_HEIGHT - 40: # Aliens reached bottom
                self.game_over = True
                self.player_lives = 0 # Instant loss
                return True
        return False

    def _spawn_wave(self):
        self.current_wave += 1
        num_aliens_to_spawn = 10
        
        alien_types = []
        if self.current_wave == 1:
            alien_types = ['red'] * 10
        elif self.current_wave == 2:
            alien_types = ['red'] * 5 + ['blue'] * 5
        elif self.current_wave == 3:
            alien_types = ['red'] * 4 + ['blue'] * 3 + ['purple'] * 3
        
        self.rng.shuffle(alien_types)

        for i in range(num_aliens_to_spawn):
            row = i // 5
            col = i % 5
            x = self.SCREEN_WIDTH * 0.2 + col * self.SCREEN_WIDTH * 0.12
            y = 50 + row * 40
            
            alien_type = alien_types[i]
            self.aliens.append({
                'pos': [x, y],
                'type': alien_type,
                'fire_timer': self.rng.integers(50, 100),
                'bob_offset': 0
            })

    def _create_starfield(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': [random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)],
                'size': random.randint(1, 2),
                'speed': random.uniform(0.1, 0.5)
            })

    def _create_explosion(self, pos, is_player=False):
        num_particles = 20 if not is_player else 40
        for _ in range(num_particles):
            self.explosions.append({
                'pos': list(pos),
                'radius': self.rng.integers(1, 5),
                'speed': self.rng.uniform(1, 3),
                'alpha': 255,
                'color': random.choice(self.COLOR_EXPLOSION)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for star in self.stars:
            star['pos'][1] += star['speed']
            if star['pos'][1] > self.SCREEN_HEIGHT:
                star['pos'][1] = 0
                star['pos'][0] = random.randint(0, self.SCREEN_WIDTH)
            pygame.draw.circle(self.screen, (200, 200, 200), (int(star['pos'][0]), int(star['pos'][1])), star['size'])

        # Player
        if self.player_lives > 0:
            p_x, p_y = int(self.player_pos[0]), int(self.player_pos[1])
            player_points = [(p_x, p_y - self.PLAYER_SIZE), (p_x - self.PLAYER_SIZE // 2, p_y + self.PLAYER_SIZE // 2), (p_x + self.PLAYER_SIZE // 2, p_y + self.PLAYER_SIZE // 2)]
            pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)
            
            # Shield effect
            if self.shield_active:
                alpha = 100 + (self.shield_timer / self.SHIELD_DURATION) * 155
                shield_color = (*self.COLOR_PLAYER_SHIELD[:3], int(alpha))
                radius = self.PLAYER_SIZE + 2
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, shield_color, (radius, radius), radius)
                self.screen.blit(temp_surf, (p_x - radius, p_y - radius))

        # Projectiles
        for p in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJECTILE, p)
        for p_data in self.alien_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_ALIEN_PROJECTILE, p_data['rect'].center, self.PROJECTILE_SIZE)

        # Aliens
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1] + alien['bob_offset'])
            color = self.COLOR_ALIEN_RED
            if alien['type'] == 'blue': color = self.COLOR_ALIEN_BLUE
            elif alien['type'] == 'purple': color = self.COLOR_ALIEN_PURPLE
            
            rect = pygame.Rect(x - self.ALIEN_SIZE, y - self.ALIEN_SIZE // 2, self.ALIEN_SIZE * 2, self.ALIEN_SIZE)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.draw.circle(self.screen, (255,255,255), (x, y - 2), 2) # Cockpit

        # Explosions
        for e in self.explosions:
            temp_surf = pygame.Surface((e['radius']*2, e['radius']*2), pygame.SRCALPHA)
            color = (*e['color'], int(e['alpha']))
            pygame.draw.circle(temp_surf, color, (e['radius'], e['radius']), e['radius'])
            self.screen.blit(temp_surf, (int(e['pos'][0] - e['radius']), int(e['pos'][1] - e['radius'])))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.player_lives):
            p_x, p_y = self.SCREEN_WIDTH - 20 - (i * (self.PLAYER_SIZE + 5)), 20
            player_points = [(p_x, p_y - self.PLAYER_SIZE//2), (p_x - self.PLAYER_SIZE // 4, p_y + self.PLAYER_SIZE // 4), (p_x + self.PLAYER_SIZE // 4, p_y + self.PLAYER_SIZE // 4)]
            pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)

        # Shield Cooldown Indicator
        if self.shield_cooldown_timer > 0:
            bar_width = 50
            bar_height = 5
            fill_width = int(bar_width * (1 - self.shield_cooldown_timer / self.SHIELD_COOLDOWN))
            pygame.draw.rect(self.screen, (50, 50, 100), (self.SCREEN_WIDTH - 65, 35, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_SHIELD, (self.SCREEN_WIDTH - 65, 35, fill_width, bar_height))

        # Game Over / Win Message
        if self.game_over:
            message = "GAME OVER" if not self.game_won else "YOU WIN!"
            end_text = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.current_wave,
            "aliens_left": len(self.aliens) + (self.TOTAL_ALIENS - self.aliens_destroyed_total - len(self.aliens)),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space after reset
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To run, you might need to unset the dummy video driver:
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Re-initialize pygame with a display
    pygame.quit()
    pygame.init()
    pygame.display.set_caption("Retro Space Shooter")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    # We need to re-assign the screen and fonts to the env instance
    # because pygame was re-initialized.
    env.screen = pygame.display.get_surface()
    env.font_small = pygame.font.Font(None, 24)
    env.font_large = pygame.font.Font(None, 50)
    
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0  # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # The observation is already the rendered frame, so we just display it.
        # We need to get the surface from the env as it's drawn there.
        surf = env.screen
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Run at 30 FPS

    env.close()