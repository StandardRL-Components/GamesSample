
# Generated: 2025-08-27T14:43:52.723148
# Source Brief: brief_00772.md
# Brief Index: 772

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys to move. Press space to fire at the nearest enemy."
    )

    # Short, user-facing description of the game
    game_description = (
        "Pilot a robot in a grid-based arena, blasting enemies to achieve total domination."
    )

    # Frames only advance when an action is received
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 10
        self.CELL_SIZE = 40
        self.MAX_STEPS = 1000
        self.PLAYER_MAX_HEALTH = 10
        self.PLAYER_ATTACK_POWER = 2
        self.ENEMY_MAX_HEALTH = 3
        self.ENEMY_ATTACK_POWER = 1
        self.NUM_ENEMIES = 5
        self.PROJECTILE_SPEED = 40 # Pixels per frame update

        # Colors
        self.COLOR_BG = (20, 20, 35)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PLAYER_FULL = (0, 255, 128)
        self.COLOR_PLAYER_MED = (255, 255, 0)
        self.COLOR_PLAYER_LOW = (255, 64, 64)
        self.COLOR_ENEMY = (255, 50, 100)
        self.COLOR_PROJECTILE = (0, 255, 255)
        self.COLOR_EXPLOSION = [(255, 255, 255), (255, 255, 0), (255, 128, 0)]
        self.COLOR_TEXT = (240, 240, 240)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_health = 0
        self.enemies = []
        self.projectiles = []
        self.explosions = []
        self.player_hit_flash = 0
        self.np_random = None
        
        # Initialize state
        self.reset()
        
        # Run self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT - 2]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_hit_flash = 0
        self.enemies = self._spawn_enemies()
        self.projectiles = []
        self.explosions = []
        
        return self._get_observation(), self._get_info()

    def _spawn_enemies(self):
        enemies = []
        occupied_positions = {tuple(self.player_pos)}
        for _ in range(self.NUM_ENEMIES):
            while True:
                pos = [
                    self.np_random.integers(0, self.GRID_WIDTH),
                    self.np_random.integers(0, self.GRID_HEIGHT - 3) # Keep them in upper part
                ]
                dist_to_player = abs(pos[0] - self.player_pos[0]) + abs(pos[1] - self.player_pos[1])
                if tuple(pos) not in occupied_positions and dist_to_player > 3:
                    enemies.append({
                        "pos": pos,
                        "health": self.ENEMY_MAX_HEALTH,
                        "hit_flash": 0
                    })
                    occupied_positions.add(tuple(pos))
                    break
        return enemies

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action
        reward = 0
        
        # --- Player Action Phase ---
        if movement > 0:
            old_pos = list(self.player_pos)
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
            
            if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
                self.player_pos = new_pos
                is_adjacent_to_enemy = any(
                    abs(e["pos"][0] - self.player_pos[0]) + abs(e["pos"][1] - self.player_pos[1]) <= 1
                    for e in self.enemies
                )
                reward += -0.2 if is_adjacent_to_enemy else 0.1

        if space_pressed:
            target = self._find_nearest_enemy()
            if target:
                # Sound: player_shoot.wav
                self.projectiles.append({
                    "start_pos": self._grid_to_pixel(self.player_pos),
                    "end_pos": self._grid_to_pixel(target["pos"]),
                    "progress": 0.0,
                    "target_enemy": target
                })

        # --- Game Logic Update Phase ---
        # Update projectiles and check for hits
        projectiles_to_remove = []
        for proj in self.projectiles:
            proj["progress"] += 0.5 # In one turn, projectile travels halfway
            if proj["progress"] >= 1.0:
                projectiles_to_remove.append(proj)
                target = proj["target_enemy"]
                if target in self.enemies: # Check if target is still alive
                    # Sound: enemy_hit.wav
                    target["health"] -= self.PLAYER_ATTACK_POWER
                    target["hit_flash"] = 3 # Flash for 3 updates
                    if target["health"] <= 0:
                        reward += 10
                        self.score += 10
                        self._create_explosion(target["pos"])
                        # Sound: explosion.wav
                        self.enemies.remove(target)
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        
        # Enemy action phase
        for enemy in self.enemies:
            dist_to_player = abs(enemy["pos"][0] - self.player_pos[0]) + abs(enemy["pos"][1] - self.player_pos[1])
            if dist_to_player <= 1:
                self.player_health -= self.ENEMY_ATTACK_POWER
                self.player_hit_flash = 3
                reward -= 5
                # Sound: player_damage.wav
        
        # --- Update effects and state ---
        self._update_effects()
        self.steps += 1
        
        # --- Termination Check ---
        terminated = False
        if not self.enemies: # Win
            reward += 100
            terminated = True
        elif self.player_health <= 0: # Loss
            reward -= 100
            terminated = True
        elif self.steps >= self.MAX_STEPS: # Timeout
            terminated = True
        
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _find_nearest_enemy(self):
        if not self.enemies:
            return None
        
        nearest_enemy = min(
            self.enemies,
            key=lambda e: (e["pos"][0] - self.player_pos[0])**2 + (e["pos"][1] - self.player_pos[1])**2
        )
        return nearest_enemy
    
    def _create_explosion(self, grid_pos):
        pixel_pos = self._grid_to_pixel(grid_pos)
        for _ in range(30): # 30 particles per explosion
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 4 + 1
            self.explosions.append({
                "pos": list(pixel_pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30),
                "color": random.choice(self.COLOR_EXPLOSION)
            })

    def _update_effects(self):
        # Update explosions
        explosions_to_remove = []
        for exp in self.explosions:
            exp["pos"][0] += exp["vel"][0]
            exp["pos"][1] += exp["vel"][1]
            exp["life"] -= 1
            if exp["life"] <= 0:
                explosions_to_remove.append(exp)
        self.explosions = [e for e in self.explosions if e not in explosions_to_remove]
        
        # Update hit flashes
        if self.player_hit_flash > 0:
            self.player_hit_flash -= 1
        for enemy in self.enemies:
            if enemy["hit_flash"] > 0:
                enemy["hit_flash"] -= 1

    def _grid_to_pixel(self, grid_pos):
        x = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return x, y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._draw_explosions()
        self._draw_enemies()
        self._draw_player()
        self._draw_projectiles()

    def _draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_player(self):
        if self.player_health <= 0: return

        px, py = self._grid_to_pixel(self.player_pos)
        size = self.CELL_SIZE * 0.7
        
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        if health_ratio > 0.6:
            color = self.COLOR_PLAYER_FULL
        elif health_ratio > 0.3:
            color = self.COLOR_PLAYER_MED
        else:
            color = self.COLOR_PLAYER_LOW
            
        if self.player_hit_flash > 0 and self.steps % 2 == 0:
            color = (255, 255, 255)

        # Body
        body_rect = pygame.Rect(px - size / 2, py - size / 2, size, size)
        pygame.draw.rect(self.screen, color, body_rect, border_radius=4)
        
        # Turret
        turret_size = size * 0.4
        turret_rect = pygame.Rect(px - turret_size / 2, py - turret_size / 2, turret_size, turret_size)
        pygame.draw.rect(self.screen, self.COLOR_GRID, turret_rect, border_radius=2)

    def _draw_enemies(self):
        size = self.CELL_SIZE * 0.6
        for enemy in self.enemies:
            px, py = self._grid_to_pixel(enemy["pos"])
            color = (255, 255, 255) if enemy["hit_flash"] > 0 else self.COLOR_ENEMY
            
            # Triangle shape for enemy
            p1 = (px, py - size/2)
            p2 = (px - size/2, py + size/2)
            p3 = (px + size/2, py + size/2)
            pygame.gfxdraw.aapolygon(self.screen, [p1,p2,p3], color)
            pygame.gfxdraw.filled_polygon(self.screen, [p1,p2,p3], color)

    def _draw_projectiles(self):
        for proj in self.projectiles:
            start_x, start_y = proj["start_pos"]
            end_x, end_y = proj["end_pos"]
            
            curr_x = int(start_x + (end_x - start_x) * proj["progress"])
            curr_y = int(start_y + (end_y - start_y) * proj["progress"])
            
            pygame.gfxdraw.aacircle(self.screen, curr_x, curr_y, 4, self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, curr_x, curr_y, 4, self.COLOR_PROJECTILE)

    def _draw_explosions(self):
        for exp in self.explosions:
            life_ratio = exp["life"] / 30.0
            radius = int((1 - life_ratio) * 10)
            alpha_color = exp["color"] + (int(life_ratio * 255),)
            
            surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, alpha_color, (radius, radius), radius)
            self.screen.blit(surf, (int(exp["pos"][0] - radius), int(exp["pos"][1] - radius)))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Health
        health_text = self.font_large.render("HEALTH:", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.SCREEN_WIDTH - 220, 10))
        
        health_bar_bg = pygame.Rect(self.SCREEN_WIDTH - 110, 15, 100, 20)
        pygame.draw.rect(self.screen, self.COLOR_GRID, health_bar_bg)
        
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        health_bar_fg = pygame.Rect(self.SCREEN_WIDTH - 110, 15, 100 * health_ratio, 20)
        
        if health_ratio > 0.6: color = self.COLOR_PLAYER_FULL
        elif health_ratio > 0.3: color = self.COLOR_PLAYER_MED
        else: color = self.COLOR_PLAYER_LOW
        pygame.draw.rect(self.screen, color, health_bar_fg)
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = not self.enemies and self.player_health > 0
            msg = "MISSION COMPLETE" if win_condition else "ROBOT DESTROYED"
            msg_render = self.font_large.render(msg, True, self.COLOR_PLAYER_FULL if win_condition else self.COLOR_PLAYER_LOW)
            msg_rect = msg_render.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_render, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "enemies_remaining": len(self.enemies),
        }
    
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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
    # Set SDL_VIDEODRIVER to "dummy" for headless execution
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    obs, info = env.reset()
    
    # Setup a visible pygame window for human play
    pygame.display.set_caption(env.game_description)
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    
    print(env.user_guide)

    while not terminated:
        movement = 0 # No-op
        space_pressed = 0
        shift_pressed = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_pressed = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_pressed = 1
        
        action = [movement, space_pressed, shift_pressed]
        
        # In a turn-based game, we only step when an action is taken
        if movement != 0 or space_pressed != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")
        
        # Render the observation to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since it's turn-based, we wait a bit to not poll keys too fast
        pygame.time.wait(50)

    env.close()