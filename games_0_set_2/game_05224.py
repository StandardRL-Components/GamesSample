
# Generated: 2025-08-28T04:21:51.662624
# Source Brief: brief_05224.md
# Brief Index: 5224

        
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

    # User-facing strings
    user_guide = "Controls: ←→ to move. Press space to fire."
    game_description = "Defend Earth from waves of descending alien invaders in this retro arcade shooter."

    # Frame advance setting
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_ALIEN = (255, 50, 50)
    COLOR_PLAYER_PROJ = (100, 150, 255)
    COLOR_ALIEN_PROJ = (255, 255, 100)
    COLOR_EXPLOSION = (255, 165, 0)
    COLOR_STAR = (200, 200, 220)
    COLOR_UI_TEXT = (255, 255, 255)

    # Game parameters
    PLAYER_SPEED = 8
    PLAYER_LIVES = 3
    PLAYER_FIRE_COOLDOWN = 6  # 5 shots per second at 30fps
    PLAYER_PROJ_SPEED = 12
    
    TOTAL_ALIENS = 50
    ALIEN_ROWS = 5
    ALIENS_PER_ROW = TOTAL_ALIENS // ALIEN_ROWS
    ALIEN_PROJ_BASE_SPEED = 3.0
    ALIEN_FIRE_CHANCE = 0.005
    ALIEN_MOVE_SPEED_INITIAL = 1.0
    ALIEN_DROP_DIST = 10
    
    MAX_STEPS = 5000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.player_lives = 0
        self.player_fire_cooldown_timer = 0
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.stars = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.aliens_destroyed_count = 0
        
        self.alien_move_direction = 1
        self.alien_move_speed = self.ALIEN_MOVE_SPEED_INITIAL
        self.alien_projectile_speed = self.ALIEN_PROJ_BASE_SPEED
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 30)
        self.player_lives = self.PLAYER_LIVES
        self.player_fire_cooldown_timer = 0
        
        self.aliens = self._create_alien_fleet()
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        
        self.aliens_destroyed_count = 0
        self.alien_move_direction = 1
        self.alien_move_speed = self.ALIEN_MOVE_SPEED_INITIAL
        self.alien_projectile_speed = self.ALIEN_PROJ_BASE_SPEED
        
        if not self.stars:
            self.stars = [
                (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT), self.np_random.integers(1, 4))
                for _ in range(150)
            ]
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
        
        reward = 0
        
        if self.game_over:
            # Still need to return a valid state
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        fire_button = action[1] == 1

        # Discourage inaction
        if movement == 0 and not fire_button:
            reward -= 0.02
        
        # --- Update Game Logic ---
        self._update_player(movement, fire_button)
        self._update_projectiles()
        self._update_aliens()
        self._update_particles()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward
        
        self.steps += 1
        
        # --- Check Termination Conditions ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if not self.aliens:  # Win condition
                reward += 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_alien_fleet(self):
        aliens = []
        for i in range(self.TOTAL_ALIENS):
            row = i // self.ALIENS_PER_ROW
            col = i % self.ALIENS_PER_ROW
            
            x = 40 + col * 50
            y = 50 + row * 40
            
            alien_rect = pygame.Rect(x, y, 30, 20)
            aliens.append(alien_rect)
        return aliens

    def _update_player(self, movement, fire_button):
        # Movement: 3=left, 4=right
        if movement == 3:
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4:
            self.player_pos.x += self.PLAYER_SPEED
        
        # Clamp player position to screen
        self.player_pos.x = max(20, min(self.SCREEN_WIDTH - 20, self.player_pos.x))
        
        # Firing
        if self.player_fire_cooldown_timer > 0:
            self.player_fire_cooldown_timer -= 1
        
        if fire_button and self.player_fire_cooldown_timer == 0:
            # sfx: player_shoot.wav
            self.player_projectiles.append(pygame.Rect(self.player_pos.x - 2, self.player_pos.y - 20, 4, 15))
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_projectiles(self):
        # Move player projectiles
        for proj in self.player_projectiles[:]:
            proj.y -= self.PLAYER_PROJ_SPEED
            if proj.bottom < 0:
                self.player_projectiles.remove(proj)
        
        # Move alien projectiles
        for proj in self.alien_projectiles[:]:
            proj.y += self.alien_projectile_speed
            if proj.top > self.SCREEN_HEIGHT:
                self.alien_projectiles.remove(proj)

    def _update_aliens(self):
        if not self.aliens:
            return

        move_down = False
        
        # Check if fleet hits side walls
        min_x = min(alien.left for alien in self.aliens)
        max_x = max(alien.right for alien in self.aliens)
        
        if (max_x + self.alien_move_speed * self.alien_move_direction > self.SCREEN_WIDTH - 10) or \
           (min_x + self.alien_move_speed * self.alien_move_direction < 10):
            self.alien_move_direction *= -1
            move_down = True
            
        # Move all aliens
        for alien in self.aliens:
            if move_down:
                alien.y += self.ALIEN_DROP_DIST
            else:
                alien.x += self.alien_move_speed * self.alien_move_direction
        
        # Random alien firing
        # Find bottom-most alien in each column
        columns = {}
        for alien in self.aliens:
            col_key = alien.x // 50
            if col_key not in columns or alien.bottom > columns[col_key].bottom:
                columns[col_key] = alien
        
        for alien in columns.values():
            if self.np_random.random() < self.ALIEN_FIRE_CHANCE:
                # sfx: alien_shoot.wav
                self.alien_projectiles.append(pygame.Rect(alien.centerx - 3, alien.bottom, 6, 12))

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs aliens
        for proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if alien.colliderect(proj):
                    # sfx: explosion.wav
                    self._spawn_explosion(alien.center, 30, self.COLOR_EXPLOSION)
                    self.aliens.remove(alien)
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    self.score += 10
                    reward += 1
                    
                    # Difficulty scaling
                    self.aliens_destroyed_count += 1
                    if self.aliens_destroyed_count % 10 == 0:
                        self.alien_projectile_speed += 0.5
                        self.alien_move_speed += 0.2
                    break
        
        # Alien projectiles vs player
        player_rect = pygame.Rect(self.player_pos.x - 15, self.player_pos.y - 10, 30, 20)
        for proj in self.alien_projectiles[:]:
            if player_rect.colliderect(proj):
                # sfx: player_hit.wav
                self.alien_projectiles.remove(proj)
                self.player_lives -= 1
                reward -= 10
                self._spawn_explosion(self.player_pos, 50, self.COLOR_PLAYER)
                if self.player_lives > 0:
                    # Brief invulnerability/respawn flash effect can be added here
                    pass
                break
        
        # Player projectiles vs alien projectiles
        for p_proj in self.player_projectiles[:]:
            for a_proj in self.alien_projectiles[:]:
                if p_proj.colliderect(a_proj):
                    # sfx: deflect.wav
                    self._spawn_explosion(p_proj.center, 5, self.COLOR_ALIEN_PROJ)
                    if p_proj in self.player_projectiles: self.player_projectiles.remove(p_proj)
                    if a_proj in self.alien_projectiles: self.alien_projectiles.remove(a_proj)
                    reward += 0.1
                    break
        
        # Aliens vs player (reaching bottom)
        for alien in self.aliens:
            if alien.bottom > self.player_pos.y - 10:
                self.player_lives = 0 # Instant loss
                reward -= 10 * (self.player_lives + 1) # Penalize for all remaining lives
                break
        
        return reward

    def _spawn_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'color': color
            })

    def _check_termination(self):
        return self.player_lives <= 0 or not self.aliens or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "aliens_remaining": len(self.aliens)
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y, size in self.stars:
            # Parallax effect
            y_shifted = (y + self.steps * 0.1 * (size/3)) % self.SCREEN_HEIGHT
            pygame.draw.circle(self.screen, self.COLOR_STAR, (int(x), int(y_shifted)), size // 2)

    def _render_game_elements(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['life'] / 4), color)

        # Render player ship if not game over
        if self.player_lives > 0:
            p = self.player_pos
            points = [(p.x, p.y - 12), (p.x - 15, p.y + 8), (p.x + 15, p.y + 8)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)
            pygame.draw.aalines(self.screen, self.COLOR_PLAYER, True, points)

        # Render aliens
        for alien in self.aliens:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN, alien, border_radius=3)
            # Simple "eyes"
            eye1 = (alien.left + 7, alien.top + 7)
            eye2 = (alien.right - 7, alien.top + 7)
            pygame.draw.circle(self.screen, self.COLOR_BG, eye1, 2)
            pygame.draw.circle(self.screen, self.COLOR_BG, eye2, 2)

        # Render projectiles
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, proj, border_radius=2)
        for proj in self.alien_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN_PROJ, proj, border_radius=3)
            
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_ui.render(f"LIVES: ", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 180, 10))
        for i in range(self.player_lives):
            p_base = pygame.Vector2(self.SCREEN_WIDTH - 100 + i * 30, 20)
            points = [(p_base.x, p_base.y - 6), (p_base.x - 7, p_base.y + 4), (p_base.x + 7, p_base.y + 4)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)

        # Alien Count
        alien_text = self.font_ui.render(f"ALIENS: {len(self.aliens)}", True, self.COLOR_UI_TEXT)
        text_rect = alien_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 15))
        self.screen.blit(alien_text, text_rect)
        
        # Game Over / Win message
        if self.game_over:
            if not self.aliens:
                msg = "MISSION COMPLETE"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption(GameEnv.game_description)
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(GameEnv.user_guide)
    
    while running:
        # --- Action mapping for human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-closing or allow reset
            pygame.time.wait(2000)
            running = False

        clock.tick(30) # Control human play speed
        
    env.close()