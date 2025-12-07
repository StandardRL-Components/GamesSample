
# Generated: 2025-08-27T22:07:15.183258
# Source Brief: brief_03022.md
# Brief Index: 3022

        
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
        "Controls: ←→ to move the ship. Hold space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-style arcade shooter. Survive 5 waves of descending aliens to win. "
        "Your ship has 3 health points. Good luck, pilot!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_EPISODE_STEPS = 10000
        self.MAX_WAVES = 5

        # Player settings
        self.PLAYER_WIDTH = 40
        self.PLAYER_HEIGHT = 20
        self.PLAYER_SPEED = 8
        self.PLAYER_FIRE_COOLDOWN = 6 # 5 shots per second

        # Bullet settings
        self.BULLET_SPEED = 12
        self.BULLET_WIDTH = 4
        self.BULLET_HEIGHT = 12

        # Alien settings
        self.ALIEN_ROWS = 4
        self.ALIEN_COLS = 8
        self.ALIEN_H_SPACING = 55
        self.ALIEN_V_SPACING = 40
        self.ALIEN_WIDTH = 30
        self.ALIEN_HEIGHT = 20
        self.ALIEN_DROP_AMOUNT = 10

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_DMG = (255, 150, 50)
        self.COLOR_BULLET = (255, 255, 100)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_BOMB = (200, 50, 255)
        self.COLOR_PARTICLE_ORANGE = (255, 165, 0)
        self.COLOR_PARTICLE_YELLOW = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_HEART = (255, 80, 80)

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)
        
        # Initialize state variables
        self.stars = []
        self._generate_stars()
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        # Player state
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        self.player_health = 3
        self.player_fire_cooldown = 0
        self.player_hit_timer = 0

        # Game object lists
        self.aliens = []
        self.bullets = []
        self.bombs = []
        self.particles = []

        # Wave and alien formation state
        self.wave = 1
        self.alien_direction = 1
        self.alien_move_down_trigger = False
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0.1 # Survival reward per frame

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            
            self._handle_input(movement, space_held)
            
            self._update_player()
            hits = self._update_bullets()
            reward += hits * 1.0 # Reward for destroying an alien
            self.score += hits * 10
            
            self._update_aliens()
            self._update_bombs()
            self._update_particles()

            if self._check_player_collisions():
                reward -= 5.0 # Penalty for getting hit
                if self.player_hit_timer <= 0:
                    self.player_health -= 1
                    self.player_hit_timer = self.FPS # 1 second of invulnerability
                    self._create_explosion(self.player_pos.x, self.player_pos.y, self.COLOR_PLAYER_DMG, 30)
                    # sfx: player_hit

            if not self.aliens and not self.win:
                self.wave += 1
                if self.wave > self.MAX_WAVES:
                    self.win = True
                    self.game_over = True
                    reward += 100.0 # Win reward
                else:
                    self._spawn_wave()
                    reward += 50.0 # Wave clear reward
                    # sfx: wave_cleared
        
        self.steps += 1
        terminated = self.player_health <= 0 or self.win or self.steps >= self.MAX_EPISODE_STEPS
        if self.player_health <= 0:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement
        if movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
        
        # Firing
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
        
        if space_held and self.player_fire_cooldown <= 0:
            bullet_rect = pygame.Rect(
                self.player_pos.x - self.BULLET_WIDTH / 2, 
                self.player_pos.y - self.PLAYER_HEIGHT,
                self.BULLET_WIDTH, self.BULLET_HEIGHT
            )
            self.bullets.append(bullet_rect)
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN
            # sfx: player_shoot

    def _update_player(self):
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_WIDTH / 2, self.WIDTH - self.PLAYER_WIDTH / 2)
        if self.player_hit_timer > 0:
            self.player_hit_timer -= 1

    def _update_bullets(self):
        hits = 0
        for b in self.bullets[:]:
            b.y -= self.BULLET_SPEED
            if b.bottom < 0:
                self.bullets.remove(b)
                continue
            
            for alien in self.aliens[:]:
                if b.colliderect(alien['rect']):
                    self._create_explosion(alien['rect'].centerx, alien['rect'].centery, self.COLOR_PARTICLE_ORANGE, 20)
                    self.aliens.remove(alien)
                    self.bullets.remove(b)
                    hits += 1
                    # sfx: alien_explosion
                    break
        return hits

    def _update_aliens(self):
        if not self.aliens:
            return

        self.alien_move_down_trigger = False
        min_x = min(a['rect'].left for a in self.aliens)
        max_x = max(a['rect'].right for a in self.aliens)

        if (max_x > self.WIDTH and self.alien_direction > 0) or \
           (min_x < 0 and self.alien_direction < 0):
            self.alien_direction *= -1
            self.alien_move_down_trigger = True
        
        for alien in self.aliens:
            if self.alien_move_down_trigger:
                alien['rect'].y += self.ALIEN_DROP_AMOUNT
            else:
                alien['rect'].x += self.alien_speed * self.alien_direction

            if alien['rect'].bottom > self.HEIGHT - 60: # Aliens reach player's general area
                self.player_health = 0
            
            if self.np_random.random() < self.alien_fire_rate:
                bomb_rect = pygame.Rect(alien['rect'].centerx - 3, alien['rect'].bottom, 6, 10)
                self.bombs.append(bomb_rect)
                # sfx: alien_shoot

    def _update_bombs(self):
        for b in self.bombs[:]:
            b.y += self.BULLET_SPEED * 0.5
            if b.top > self.HEIGHT:
                self.bombs.remove(b)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_player_collisions(self):
        if self.player_hit_timer > 0:
            return False
            
        player_rect = pygame.Rect(
            self.player_pos.x - self.PLAYER_WIDTH / 2, 
            self.player_pos.y - self.PLAYER_HEIGHT / 2, 
            self.PLAYER_WIDTH, self.PLAYER_HEIGHT
        )
        for bomb in self.bombs[:]:
            if player_rect.colliderect(bomb):
                self.bombs.remove(bomb)
                return True
        return False
    
    def _spawn_wave(self):
        self.alien_speed = 0.5 + (self.wave - 1) * 0.1
        self.alien_fire_rate = 0.002 + (self.wave - 1) * 0.001
        
        start_x = (self.WIDTH - (self.ALIEN_COLS * self.ALIEN_H_SPACING)) / 2 + self.ALIEN_H_SPACING / 2
        start_y = 50

        for row in range(self.ALIEN_ROWS):
            for col in range(self.ALIEN_COLS):
                x = start_x + col * self.ALIEN_H_SPACING
                y = start_y + row * self.ALIEN_V_SPACING
                alien_rect = pygame.Rect(x, y, self.ALIEN_WIDTH, self.ALIEN_HEIGHT)
                self.aliens.append({'rect': alien_rect})

    def _create_explosion(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': pygame.Vector2(x, y), 'vel': vel, 'life': life, 'color': color})

    def _generate_stars(self):
        for _ in range(200):
            self.stars.append({
                'pos': pygame.Vector2(self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                'size': self.np_random.random() * 1.5 + 0.5,
                'brightness': self.np_random.integers(50, 150)
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for star in self.stars:
            brightness = star['brightness']
            if self.np_random.random() < 0.01: # Twinkle
                brightness = self.np_random.integers(50, 200)
            color = (brightness, brightness, brightness)
            pygame.draw.circle(self.screen, color, (int(star['pos'].x), int(star['pos'].y)), int(star['size']))

        # Player
        if self.player_health > 0:
            is_hit = self.player_hit_timer > 0 and (self.player_hit_timer // 2) % 2 == 0
            if not is_hit:
                p_x, p_y = self.player_pos.x, self.player_pos.y
                w, h = self.PLAYER_WIDTH, self.PLAYER_HEIGHT
                points = [
                    (p_x, p_y - h / 2),
                    (p_x - w / 2, p_y + h / 2),
                    (p_x + w / 2, p_y + h / 2)
                ]
                pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)
                pygame.draw.aaline(self.screen, self.COLOR_PLAYER, points[0], points[1])
                pygame.draw.aaline(self.screen, self.COLOR_PLAYER, points[1], points[2])
                pygame.draw.aaline(self.screen, self.COLOR_PLAYER, points[2], points[0])

        # Bullets
        for b in self.bullets:
            pygame.draw.rect(self.screen, self.COLOR_BULLET, b, border_radius=3)
        
        # Aliens
        for alien in self.aliens:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN, alien['rect'], border_radius=4)
        
        # Bombs
        for b in self.bombs:
            pygame.gfxdraw.filled_circle(self.screen, b.centerx, b.centery, b.width // 2, self.COLOR_BOMB)
            pygame.gfxdraw.aacircle(self.screen, b.centerx, b.centery, b.width // 2, self.COLOR_BOMB)

        # Particles
        for p in self.particles:
            life_ratio = p['life'] / 30.0
            size = int(life_ratio * 5)
            if size > 0:
                alpha = int(life_ratio * 255)
                color = p['color']
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), size, (*color, alpha))
    
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_ui.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Health
        for i in range(self.player_health):
            heart_pos = (self.WIDTH / 2 - 30 + i * 30, 15)
            pygame.draw.circle(self.screen, self.COLOR_HEART, (heart_pos[0] + 5, heart_pos[1] + 5), 7)
            pygame.draw.circle(self.screen, self.COLOR_HEART, (heart_pos[0] - 5, heart_pos[1] + 5), 7)
            pygame.draw.polygon(self.screen, self.COLOR_HEART, [(heart_pos[0]-12, heart_pos[1]+5), (heart_pos[0]+12, heart_pos[1]+5), (heart_pos[0], heart_pos[1]+18)])

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_ALIEN
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player_health
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Arcade Shooter")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated:
            # Get keyboard inputs
            keys = pygame.key.get_pressed()
            
            # Map keys to action space
            movement = 0 # no-op
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        else:
            # If the game is over, you might want to show a restart message
            # For now, we just wait for the user to close the window
            pass

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()