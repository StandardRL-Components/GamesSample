
# Generated: 2025-08-28T05:13:10.195178
# Source Brief: brief_05496.md
# Brief Index: 5496

        
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

    user_guide = (
        "Controls: Arrow keys to move. Hold Space to fire. Avoid alien ships and their projectiles."
    )

    game_description = (
        "Defend Earth from waves of descending alien invaders in this retro top-down arcade shooter."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PLAYER_SPEED = 6
        self.BULLET_SPEED = 10
        self.PLAYER_FIRE_COOLDOWN_MAX = 8  # Allows ~4 shots per second at 30fps
        self.MAX_STEPS = 10000
        self.NUM_WAVES = 5
        self.PLAYER_LIVES_START = 3

        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_THRUSTER = (255, 255, 0)
        self.COLOR_ALIEN = (255, 64, 64)
        self.COLOR_PLAYER_BULLET = (128, 255, 255)
        self.COLOR_ALIEN_BULLET = (255, 128, 255)
        self.COLOR_STAR = (200, 200, 220)
        self.EXPLOSION_COLORS = [(255, 255, 0), (255, 128, 0), (255, 0, 0)]
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_UI_GAMEOVER = (255, 0, 0)
        self.COLOR_UI_WIN = (0, 255, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_gameover = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- Game State Variables (initialized in reset) ---
        self.player_pos = None
        self.player_lives = None
        self.player_fire_cooldown = None
        self.player_bullets = None
        self.aliens = None
        self.alien_bullets = None
        self.alien_move_direction = None
        self.alien_move_timer = None
        self.alien_speed = None
        self.alien_fire_rate = None
        self.stars = None
        self.particles = None
        self.steps = None
        self.score = None
        self.wave = None
        self.game_over = None
        self.win = None
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False
        self.win = False
        
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.player_lives = self.PLAYER_LIVES_START
        self.player_fire_cooldown = 0
        
        self.player_bullets = []
        self.alien_bullets = []
        self.aliens = []
        self.particles = []
        
        self._spawn_wave()
        
        if self.stars is None: # Generate stars only once for performance
            self.stars = [
                (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
                for _ in range(150)
            ]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for each step to encourage efficiency

        if not self.game_over:
            # --- Unpack Action ---
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            
            # --- Update Game Logic ---
            self._handle_player_input(movement, space_held)
            self._update_bullets()
            self._update_aliens()
            self._update_particles()
            
            # --- Handle Collisions & Get Rewards ---
            collision_rewards = self._handle_collisions()
            reward += collision_rewards

            # --- Check for Wave Completion ---
            if not self.aliens and not self.win:
                reward += 10.0  # Wave clear bonus
                self.wave += 1
                if self.wave > self.NUM_WAVES:
                    self.win = True
                    self.game_over = True
                    reward += 100.0 # Victory bonus
                else:
                    # Sound: Wave clear
                    self._spawn_wave()
        
        # --- Update Step Counter & Check Termination ---
        self.steps += 1
        terminated = self._check_termination()
        if terminated and not self.game_over:
             self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held):
        # Player Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED # Up
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED # Down
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED # Left
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED # Right

        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)
        self.player_pos[1] = np.clip(self.player_pos[1], self.HEIGHT - 100, self.HEIGHT - 20)

        # Player Firing
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
            
        if space_held and self.player_fire_cooldown <= 0:
            # Sound: Player shoot
            bullet_pos = [self.player_pos[0], self.player_pos[1] - 15]
            self.player_bullets.append(pygame.Rect(bullet_pos[0] - 2, bullet_pos[1], 4, 10))
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN_MAX

    def _update_bullets(self):
        # Move player bullets
        for bullet in self.player_bullets[:]:
            bullet.y -= self.BULLET_SPEED
            if bullet.bottom < 0:
                self.player_bullets.remove(bullet)

        # Move alien bullets
        for bullet in self.alien_bullets[:]:
            bullet.y += self.BULLET_SPEED / 2
            if bullet.top > self.HEIGHT:
                self.alien_bullets.remove(bullet)

    def _update_aliens(self):
        self.alien_move_timer -= 1
        move_horizontally = False
        if self.alien_move_timer <= 0:
            move_horizontally = True
            self.alien_move_timer = max(5, 30 - self.wave * 4)

        drop_down = False
        for alien in self.aliens:
            if move_horizontally:
                alien['rect'].x += self.alien_speed * self.alien_move_direction
            
            if (alien['rect'].right > self.WIDTH - 10 and self.alien_move_direction > 0) or \
               (alien['rect'].left < 10 and self.alien_move_direction < 0):
                drop_down = True

            if self.np_random.random() < self.alien_fire_rate:
                # Sound: Alien shoot
                bullet_pos = [alien['rect'].centerx, alien['rect'].bottom]
                self.alien_bullets.append(pygame.Rect(bullet_pos[0] - 2, bullet_pos[1], 4, 10))

        if drop_down:
            self.alien_move_direction *= -1
            for alien in self.aliens:
                alien['rect'].y += 15

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        for bullet in self.player_bullets[:]:
            for alien in self.aliens[:]:
                if alien['rect'].colliderect(bullet):
                    # Sound: Alien explosion
                    self._create_explosion(alien['rect'].center, self.EXPLOSION_COLORS)
                    self.aliens.remove(alien)
                    if bullet in self.player_bullets: self.player_bullets.remove(bullet)
                    self.score += 100
                    reward += 1.0
                    break
        
        player_rect = pygame.Rect(self.player_pos[0] - 12, self.player_pos[1] - 10, 24, 20)
        
        for bullet in self.alien_bullets[:]:
            if player_rect.colliderect(bullet):
                self.alien_bullets.remove(bullet)
                self._hit_player()
                break

        for alien in self.aliens:
            if player_rect.colliderect(alien['rect']):
                self._hit_player()
                # Sound: Alien explosion
                self._create_explosion(alien['rect'].center, self.EXPLOSION_COLORS)
                self.aliens.remove(alien)
                break
        
        return reward

    def _hit_player(self):
        # Sound: Player explosion
        self._create_explosion(self.player_pos, self.EXPLOSION_COLORS)
        self.player_lives -= 1
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        if self.player_lives <= 0:
            self.game_over = True
    
    def _check_termination(self):
        return self.player_lives <= 0 or self.steps >= self.MAX_STEPS or self.win

    def _spawn_wave(self):
        self.aliens.clear()
        self.alien_bullets.clear()
        self.alien_move_direction = 1
        self.alien_move_timer = 30
        
        self.alien_speed = 1.0 + (self.wave - 1) * 0.5
        self.alien_fire_rate = 0.0005 + (self.wave - 1) * 0.0002

        num_rows = 4
        num_cols = 10
        for row in range(num_rows):
            for col in range(num_cols):
                x = 40 + col * 55
                y = 40 + row * 40
                self.aliens.append({'rect': pygame.Rect(x, y, 30, 20)})

    def _create_explosion(self, position, colors):
        for _ in range(30):
            self.particles.append({
                'pos': list(position),
                'vel': [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)],
                'life': self.np_random.integers(15, 30),
                'color': random.choice(colors),
                'radius': self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size / 2)

        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 30.0))
            s = self.screen.copy()
            s.set_colorkey((0,0,0))
            s.set_alpha(alpha)
            pygame.gfxdraw.filled_circle(s, int(p['pos'][0]), int(p['pos'][1]), int(p['radius'] * (p['life'] / 15.0)), p['color'])
            self.screen.blit(s, (0,0))

        for bullet in self.alien_bullets:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN_BULLET, bullet)

        for bullet in self.player_bullets:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_BULLET, bullet)

        for alien in self.aliens:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN, alien['rect'], border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_BG, alien['rect'].inflate(-8, -8), border_radius=3)

        if self.player_lives > 0:
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            player_points = [(px, py - 12), (px - 15, py + 10), (px + 15, py + 10)]
            pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)
            thruster_points = [(px, py + 15), (px - 5, py + 10), (px + 5, py + 10)]
            pygame.gfxdraw.filled_polygon(self.screen, thruster_points, self.COLOR_PLAYER_THRUSTER)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        wave_text = self.font_main.render(f"WAVE: {self.wave}/{self.NUM_WAVES}", True, self.COLOR_UI)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        for i in range(self.player_lives):
            px, py = 25 + i * 35, self.HEIGHT - 25
            points = [(px, py - 8), (px - 10, py + 6), (px + 10, py + 6)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_UI_WIN if self.win else self.COLOR_UI_GAMEOVER
            end_text = self.font_gameover.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "lives": self.player_lives,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import os
    is_headless = "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy"

    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    if is_headless:
        print("Running in headless mode. No interactive demo.")
        env.close()
        exit()

    try:
        pygame.display.init()
        pygame.display.set_caption("Arcade Shooter")
        display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    except pygame.error:
        print("No display available. Cannot run interactive demo.")
        env.close()
        exit()

    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        env.clock.tick(30)

    print(f"Game Over! Final Info: {info}")
    pygame.time.wait(2000)
    env.close()