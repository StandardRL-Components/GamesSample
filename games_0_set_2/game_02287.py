
# Generated: 2025-08-28T04:19:44.583842
# Source Brief: brief_02287.md
# Brief Index: 2287

        
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

    # Short, user-facing control string
    user_guide = "Controls: ←→ to move. Hold Space to fire."

    # Short, user-facing description of the game
    game_description = (
        "A retro-futuristic arcade shooter. Defend your base from waves of descending alien invaders."
    )

    # Frames auto-advance at 30fps
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (16, 16, 24)  # Dark blue-gray
    COLOR_GRID = (32, 32, 48)
    COLOR_PLAYER = (0, 255, 255)  # Cyan
    COLOR_PLAYER_GLOW = (0, 128, 128)
    COLOR_P_PROJECTILE = (127, 255, 0)  # Chartreuse
    COLOR_A_PROJECTILE = (255, 220, 0)  # Gold
    ALIEN_COLORS = [(255, 65, 54), (255, 133, 27), (255, 100, 100)] # Red, Orange, Pink
    PARTICLE_COLORS = [(255, 255, 255), (255, 0, 255), (128, 0, 128)] # White, Magenta, Purple

    # Player settings
    PLAYER_Y_POS = 360
    PLAYER_WIDTH = 30
    PLAYER_HEIGHT = 15
    PLAYER_SPEED = 8
    PLAYER_FIRE_COOLDOWN = 6  # 5 shots per second

    # Alien settings
    ALIEN_COLS = 10
    ALIEN_ROWS = 4
    ALIEN_SPACING = 40
    ALIEN_SIZE = 16
    ALIEN_BASE_SPEED = 0.5
    ALIEN_DROP_HEIGHT = 15
    ALIEN_BASE_FIRE_CHANCE = 0.001
    
    # Projectile settings
    PROJECTILE_SPEED = 12
    PROJECTILE_WIDTH = 4
    PROJECTILE_HEIGHT = 12

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_lives = None
        self.player_fire_cooldown_timer = None
        self.aliens = None
        self.player_projectiles = None
        self.alien_projectiles = None
        self.particles = None
        self.alien_move_direction = None
        self.alien_move_down_flag = None
        self.aliens_destroyed_count = None
        self.alien_speed_multiplier = None
        self.alien_fire_rate_multiplier = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.player_pos = [self.SCREEN_WIDTH / 2, self.PLAYER_Y_POS]
        self.player_lives = 3
        self.player_fire_cooldown_timer = 0
        
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []

        self._spawn_aliens()
        self.alien_move_direction = 1
        self.alien_move_down_flag = False
        self.aliens_destroyed_count = 0
        self.alien_speed_multiplier = 1.0
        self.alien_fire_rate_multiplier = 1.0

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = -0.01  # Small penalty for each step to encourage efficiency

        # --- Handle Actions ---
        movement, space_held, _ = action
        
        # Player Movement
        if movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_WIDTH / 2, self.SCREEN_WIDTH - self.PLAYER_WIDTH / 2)

        # Player Firing
        if space_held and self.player_fire_cooldown_timer <= 0:
            self._fire_player_projectile()
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN

        # --- Update Game State ---
        self._update_timers()
        self._update_projectiles()
        self._update_aliens()
        self._update_particles()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100
            else: # Loss or time limit
                reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_aliens(self):
        self.aliens = []
        grid_width = self.ALIEN_COLS * self.ALIEN_SPACING
        start_x = (self.SCREEN_WIDTH - grid_width) / 2 + self.ALIEN_SPACING / 2
        for row in range(self.ALIEN_ROWS):
            for col in range(self.ALIEN_COLS):
                x = start_x + col * self.ALIEN_SPACING
                y = 50 + row * self.ALIEN_SPACING
                self.aliens.append({
                    "pos": [x, y],
                    "color": self.ALIEN_COLORS[row % len(self.ALIEN_COLORS)],
                    "size": self.ALIEN_SIZE,
                })

    def _fire_player_projectile(self):
        # sound: player_shoot.wav
        pos = [self.player_pos[0], self.player_pos[1] - self.PLAYER_HEIGHT]
        self.player_projectiles.append(pygame.Rect(pos[0] - self.PROJECTILE_WIDTH / 2, pos[1], self.PROJECTILE_WIDTH, self.PROJECTILE_HEIGHT))

    def _update_timers(self):
        if self.player_fire_cooldown_timer > 0:
            self.player_fire_cooldown_timer -= 1

    def _update_projectiles(self):
        self.player_projectiles = [p for p in self.player_projectiles if p.y > -self.PROJECTILE_HEIGHT]
        for p in self.player_projectiles:
            p.y -= self.PROJECTILE_SPEED

        self.alien_projectiles = [p for p in self.alien_projectiles if p.y < self.SCREEN_HEIGHT]
        for p in self.alien_projectiles:
            p.y += self.PROJECTILE_SPEED / 2

    def _update_aliens(self):
        if not self.aliens:
            return

        move_x = self.ALIEN_BASE_SPEED * self.alien_speed_multiplier * self.alien_move_direction
        move_y = 0

        if self.alien_move_down_flag:
            move_y = self.ALIEN_DROP_HEIGHT
            self.alien_move_direction *= -1
            self.alien_move_down_flag = False

        edge_hit = False
        for alien in self.aliens:
            alien["pos"][0] += move_x
            alien["pos"][1] += move_y
            if not edge_hit and (alien["pos"][0] < self.ALIEN_SIZE or alien["pos"][0] > self.SCREEN_WIDTH - self.ALIEN_SIZE):
                edge_hit = True

            # Alien firing logic
            fire_chance = self.ALIEN_BASE_FIRE_CHANCE * self.alien_fire_rate_multiplier
            if self.np_random.random() < fire_chance:
                # sound: alien_shoot.wav
                pos = alien["pos"]
                self.alien_projectiles.append(pygame.Rect(pos[0] - 2, pos[1], 4, 8))
        
        if edge_hit:
            self.alien_move_down_flag = True

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        projectiles_to_remove = []
        aliens_to_remove = []
        for i, p_rect in enumerate(self.player_projectiles):
            for j, alien in enumerate(self.aliens):
                if j in aliens_to_remove: continue
                alien_rect = pygame.Rect(alien["pos"][0] - alien["size"]/2, alien["pos"][1] - alien["size"]/2, alien["size"], alien["size"])
                if p_rect.colliderect(alien_rect):
                    # sound: explosion.wav
                    self._create_explosion(alien["pos"], 30, alien["color"])
                    aliens_to_remove.append(j)
                    if i not in projectiles_to_remove:
                        projectiles_to_remove.append(i)
                    reward += 1
                    self.score += 10
                    self.aliens_destroyed_count += 1
                    if self.aliens_destroyed_count > 0 and self.aliens_destroyed_count % 10 == 0:
                        self.alien_speed_multiplier *= 1.05
                        self.alien_fire_rate_multiplier *= 1.05

        self.player_projectiles = [p for i, p in enumerate(self.player_projectiles) if i not in projectiles_to_remove]
        self.aliens = [a for i, a in enumerate(self.aliens) if i not in sorted(aliens_to_remove, reverse=True)]

        # Alien projectiles vs Player
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_WIDTH / 2, self.player_pos[1] - self.PLAYER_HEIGHT / 2, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        projectiles_to_remove = []
        for i, p_rect in enumerate(self.alien_projectiles):
            if p_rect.colliderect(player_rect):
                # sound: player_hit.wav
                if i not in projectiles_to_remove:
                    projectiles_to_remove.append(i)
                self.player_lives -= 1
                reward -= 10
                self._create_explosion(self.player_pos, 50, self.COLOR_PLAYER)
                if self.player_lives <= 0:
                    self.game_over = True

        self.alien_projectiles = [p for i, p in enumerate(self.alien_projectiles) if i not in projectiles_to_remove]
        
        return reward

    def _check_termination(self):
        if self.game_over: # Player lives <= 0
            return True
        if not self.aliens: # Win condition
            self.game_over = True
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS: # Time limit
            self.game_over = True
            return True
        # Check if any alien reached the bottom
        for alien in self.aliens:
            if alien["pos"][1] > self.PLAYER_Y_POS - 20:
                self.game_over = True
                self.player_lives = 0
                return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_aliens()
        self._render_player()
        self._render_projectiles()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.player_lives}
        
    def _create_explosion(self, pos, count, base_color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(10, 25),
                'color': random.choice(self.PARTICLE_COLORS)
            })

    # --- Rendering Methods ---
    def _render_background(self):
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

    def _render_player(self):
        if self.player_lives > 0:
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            hw, hh = self.PLAYER_WIDTH / 2, self.PLAYER_HEIGHT / 2
            points = [(px, py - hh), (px - hw, py + hh), (px + hw, py + hh)]
            
            # Glow effect
            for i in range(10, 0, -2):
                alpha = 80 - i * 8
                pygame.gfxdraw.filled_polygon(self.screen, [(p[0], p[1]+i/4) for p in points], (*self.COLOR_PLAYER_GLOW, alpha))

            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_aliens(self):
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            size = int(alien['size'])
            pulse = (math.sin(self.steps * 0.2) + 1) / 4 + 0.75 # 0.75 to 1.25
            r = int(size / 2 * pulse)
            
            pygame.gfxdraw.aacircle(self.screen, x, y, r, alien['color'])
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, alien['color'])

    def _render_projectiles(self):
        for p in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_P_PROJECTILE, p)
            # glow
            p_glow = p.copy()
            p_glow.inflate_ip(4,4)
            pygame.draw.rect(self.screen, self.COLOR_P_PROJECTILE, p_glow, 1, border_radius=2)
            
        for p in self.alien_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_A_PROJECTILE, p, border_radius=2)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * 15)))
            color_with_alpha = (*p['color'], alpha)
            size = max(1, int(p['life'] / 5))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color_with_alpha)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_ui.render("LIVES:", True, (255, 255, 255))
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 180, 10))
        for i in range(self.player_lives):
            px = self.SCREEN_WIDTH - 100 + i * 30
            py = 18
            points = [(px, py-7), (px-10, py+7), (px+10, py+7)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
            
    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))
        
        message = "YOU WIN" if self.win else "GAME OVER"
        color = (0, 255, 128) if self.win else (255, 0, 64)
        
        text_surf = self.font_game_over.render(message, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def close(self):
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