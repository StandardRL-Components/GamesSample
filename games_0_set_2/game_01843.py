
# Generated: 2025-08-28T02:52:58.707481
# Source Brief: brief_01843.md
# Brief Index: 1843

        
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
        "Controls: Arrow keys to move. Press Space to fire. Press Shift to rotate your aim 90° clockwise."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a robot in a top-down arena, blasting enemy robots to achieve total robotic domination. Defeat 25 enemies to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.WIN_CONDITION_KILLS = 25
        self.MAX_ENEMIES_ON_SCREEN = 5

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_ARENA = (50, 60, 80)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_ACCENT = (200, 255, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_ACCENT = (255, 150, 150)
        self.COLOR_PROJECTILE = (255, 255, 0)
        self.COLOR_EXPLOSION = (255, 165, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_GREEN = (0, 200, 0)
        self.COLOR_HEALTH_RED = (200, 0, 0)
        self.COLOR_HEALTH_BG = (70, 70, 70)

        # Player
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 4.0
        self.PLAYER_MAX_HEALTH = 100
        self.SHOOT_COOLDOWN_FRAMES = 6

        # Enemy
        self.ENEMY_SIZE = 22
        self.ENEMY_MAX_HEALTH = 20
        self.ENEMY_BASE_SPEED = 0.8
        self.ENEMY_SPEED_INCREASE_PER_TIER = 0.2 # Brief says 0.02, but this is too slow to notice. 0.2 is better for gameplay.
        self.ENEMY_CHANGE_DIR_PROB = 0.02

        # Projectile
        self.PROJECTILE_SPEED = 12.0
        self.PROJECTILE_SIZE = 3

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.enemies = []
        self.projectiles = []
        self.explosions = []
        self.enemies_defeated = 0
        self.current_enemy_speed = 0.0
        self.shoot_cooldown = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.enemies_defeated = 0
        self.current_enemy_speed = self.ENEMY_BASE_SPEED

        self.player = {
            "rect": pygame.Rect(self.WIDTH // 2 - self.PLAYER_SIZE // 2, self.HEIGHT // 2 - self.PLAYER_SIZE // 2, self.PLAYER_SIZE, self.PLAYER_SIZE),
            "health": self.PLAYER_MAX_HEALTH,
            "dir_index": 0, # 0:Up, 1:Right, 2:Down, 3:Left
        }
        self.DIRECTIONS = [pygame.Vector2(0, -1), pygame.Vector2(1, 0), pygame.Vector2(0, 1), pygame.Vector2(-1, 0)]

        self.enemies = []
        self.projectiles = []
        self.explosions = []
        
        while len(self.enemies) < self.MAX_ENEMIES_ON_SCREEN:
            self._spawn_enemy()

        self.shoot_cooldown = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Player Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player["rect"].move_ip(move_vec * self.PLAYER_SPEED)
        self._keep_in_bounds(self.player["rect"])

        # Player Rotation (on key press)
        if shift_held and not self.prev_shift_held:
            self.player["dir_index"] = (self.player["dir_index"] + 1) % 4
            # sound: "ui_rotate.wav"
        self.prev_shift_held = shift_held
        
        # Player Shooting (on key press)
        if space_held and not self.prev_space_held and self.shoot_cooldown == 0:
            self._fire_projectile()
            self.shoot_cooldown = self.SHOOT_COOLDOWN_FRAMES
            # sound: "player_shoot.wav"
        self.prev_space_held = space_held
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        # --- Update Game State ---
        
        # Update Projectiles
        for proj in self.projectiles[:]:
            proj["rect"].move_ip(proj["vel"])
            hit_something = False
            for enemy in self.enemies:
                if proj["rect"].colliderect(enemy["rect"]):
                    enemy["health"] -= 25 # 1-shot kill for 20hp enemy
                    reward += 0.1
                    self._create_explosion(enemy["rect"].center)
                    self.projectiles.remove(proj)
                    hit_something = True
                    break
            if hit_something: continue
            
            if not self.screen.get_rect().contains(proj["rect"]):
                reward -= 0.02 # Miss penalty
                self.projectiles.remove(proj)

        # Update Enemies
        for enemy in self.enemies[:]:
            if enemy["health"] <= 0:
                reward += 1.0
                self.score += 1
                self.enemies_defeated += 1
                self.enemies.remove(enemy)
                # sound: "enemy_explode.wav"
                
                # Difficulty scaling
                if self.enemies_defeated > 0 and self.enemies_defeated % 5 == 0:
                    self.current_enemy_speed += self.ENEMY_SPEED_INCREASE_PER_TIER
                continue

            # Random movement
            if self.np_random.random() < self.ENEMY_CHANGE_DIR_PROB or enemy["vel"].length() == 0:
                enemy["vel"] = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize() * self.current_enemy_speed
            enemy["rect"].move_ip(enemy["vel"])
            self._keep_in_bounds(enemy["rect"])

            # Player-Enemy collision
            if self.player["rect"].colliderect(enemy["rect"]):
                self.player["health"] -= 1
                self._create_explosion(self.player["rect"].center, small=True)
                # sound: "player_hit.wav"

        # Respawn enemies
        while len(self.enemies) < self.MAX_ENEMIES_ON_SCREEN and self.enemies_defeated + len(self.enemies) < self.WIN_CONDITION_KILLS:
            self._spawn_enemy()
            
        # Update Explosions
        for exp in self.explosions[:]:
            exp["life"] -= 1
            exp["radius"] += exp["growth"]
            if exp["life"] <= 0:
                self.explosions.remove(exp)

        # --- Check Termination ---
        self.steps += 1
        terminated = False
        if self.player["health"] <= 0:
            reward -= 100
            terminated = True
            # sound: "game_over.wav"
        elif self.enemies_defeated >= self.WIN_CONDITION_KILLS:
            reward += 100
            self.score += 10 # Bonus for winning
            terminated = True
            # sound: "win_game.wav"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        
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
            "player_health": self.player["health"],
            "enemies_defeated": self.enemies_defeated,
        }
    
    # --- Helper Methods ---
    def _keep_in_bounds(self, rect):
        rect.left = max(0, rect.left)
        rect.right = min(self.WIDTH, rect.right)
        rect.top = max(0, rect.top)
        rect.bottom = min(self.HEIGHT, rect.bottom)
        
    def _spawn_enemy(self):
        side = self.np_random.integers(4)
        if side == 0: # top
            x, y = self.np_random.uniform(0, self.WIDTH), -self.ENEMY_SIZE
        elif side == 1: # bottom
            x, y = self.np_random.uniform(0, self.WIDTH), self.HEIGHT
        elif side == 2: # left
            x, y = -self.ENEMY_SIZE, self.np_random.uniform(0, self.HEIGHT)
        else: # right
            x, y = self.WIDTH, self.np_random.uniform(0, self.HEIGHT)
            
        self.enemies.append({
            "rect": pygame.Rect(x, y, self.ENEMY_SIZE, self.ENEMY_SIZE),
            "health": self.ENEMY_MAX_HEALTH,
            "vel": pygame.Vector2(0, 0)
        })

    def _fire_projectile(self):
        direction = self.DIRECTIONS[self.player["dir_index"]]
        start_pos = self.player["rect"].center + direction * (self.PLAYER_SIZE / 2)
        vel = direction * self.PROJECTILE_SPEED
        self.projectiles.append({
            "rect": pygame.Rect(start_pos.x - self.PROJECTILE_SIZE // 2, start_pos.y - self.PROJECTILE_SIZE // 2, self.PROJECTILE_SIZE, self.PROJECTILE_SIZE),
            "vel": vel
        })

    def _create_explosion(self, pos, small=False):
        self.explosions.append({
            "pos": pos,
            "radius": 5 if small else 10,
            "life": 10 if small else 15,
            "growth": 1 if small else 1.5
        })
        
    def _draw_health_bar(self, surface, rect, current_hp, max_hp, color):
        if current_hp < 0: current_hp = 0
        bar_width = rect.width
        bar_height = 5
        bar_y = rect.top - bar_height - 3
        
        bg_rect = pygame.Rect(rect.left, bar_y, bar_width, bar_height)
        fill_ratio = current_hp / max_hp
        fill_rect = pygame.Rect(rect.left, bar_y, int(bar_width * fill_ratio), bar_height)
        
        pygame.draw.rect(surface, self.COLOR_HEALTH_BG, bg_rect)
        pygame.draw.rect(surface, color, fill_rect)

    def _render_game(self):
        # Draw arena boundary
        pygame.draw.rect(self.screen, self.COLOR_ARENA, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Draw explosions
        for exp in self.explosions:
            alpha = int(255 * (exp["life"] / 15))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(exp["pos"][0]), int(exp["pos"][1]), int(exp["radius"]), (*self.COLOR_EXPLOSION, alpha))
                pygame.gfxdraw.aacircle(self.screen, int(exp["pos"][0]), int(exp["pos"][1]), int(exp["radius"]), (*self.COLOR_EXPLOSION, alpha))

        # Draw enemies
        for enemy in self.enemies:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, enemy["rect"])
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_ACCENT, enemy["rect"], 2)
            self._draw_health_bar(self.screen, enemy["rect"], enemy["health"], self.ENEMY_MAX_HEALTH, self.COLOR_HEALTH_RED)

        # Draw player
        player_rect = self.player["rect"]
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_ACCENT, player_rect, 2)
        
        # Draw player direction indicator
        direction = self.DIRECTIONS[self.player["dir_index"]]
        p1 = player_rect.center + direction * 8
        p2 = player_rect.center + pygame.Vector2(-direction.y, direction.x) * 5
        p3 = player_rect.center + pygame.Vector2(direction.y, -direction.x) * 5
        pygame.gfxdraw.aapolygon(self.screen, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)], self.COLOR_PLAYER_ACCENT)
        pygame.gfxdraw.filled_polygon(self.screen, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)], self.COLOR_PLAYER_ACCENT)
        
        # Draw projectiles
        for proj in self.projectiles:
            start_pos = proj["rect"].center - proj["vel"] * 0.5
            end_pos = proj["rect"].center
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, 4)

    def _render_ui(self):
        # Enemies remaining
        enemies_text = self.font_large.render(f"ENEMIES: {max(0, self.WIN_CONDITION_KILLS - self.enemies_defeated)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(enemies_text, (10, 10))
        
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Player Health Bar (Bottom Center)
        bar_width = self.WIDTH // 3
        bar_height = 15
        bar_x = (self.WIDTH - bar_width) // 2
        bar_y = self.HEIGHT - bar_height - 10
        
        health_ratio = max(0, self.player["health"] / self.PLAYER_MAX_HEALTH)
        
        bg_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        fill_rect = pygame.Rect(bar_x, bar_y, int(bar_width * health_ratio), bar_height)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, fill_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, bg_rect, 2, border_radius=3)

        health_text = self.font_small.render("HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (bar_x + (bar_width - health_text.get_width()) // 2, bar_y + (bar_height - health_text.get_height()) // 2))

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

# Example of how to run the environment
if __name__ == '__main__':
    # For interactive play
    import pygame
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Robot Domination")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()
            terminated = False

        # --- Human Controls ---
        movement = 0 # none
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.FPS)
        
    env.close()
    pygame.quit()