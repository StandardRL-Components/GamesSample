
# Generated: 2025-08-27T16:58:15.893767
# Source Brief: brief_01388.md
# Brief Index: 1388

        
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
        "Controls: Arrow keys to move. Hold Space to fire. Shift has no effect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a robot in a top-down arena, blasting enemies to achieve total annihilation."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1500
        self.NUM_ENEMIES = 15

        # --- Colors ---
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_PLAYER = (255, 64, 64)
        self.COLOR_PLAYER_GLOW = (255, 100, 100)
        self.COLOR_ENEMY = (64, 128, 255)
        self.COLOR_ENEMY_GLOW = (100, 150, 255)
        self.COLOR_PLAYER_PROJ = (255, 255, 0)
        self.COLOR_ENEMY_PROJ = (255, 0, 255)
        self.COLOR_EXPLOSION = (255, 165, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (0, 255, 0)
        self.COLOR_HEALTH_BAR_BG = (100, 100, 100)
        self.COLOR_ARENA_BORDER = (200, 200, 200)

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_health = None
        self.player_fire_cooldown = None
        self.last_move_direction = None
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.explosions = []
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = 100.0
        self.player_fire_cooldown = 0
        self.last_move_direction = pygame.Vector2(0, -1) # Default to firing up

        # Lists for dynamic objects
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.explosions = []

        # Spawn enemies
        for i in range(self.NUM_ENEMIES):
            padding = 50
            orbit_center = pygame.Vector2(
                self.np_random.uniform(padding, self.WIDTH - padding),
                self.np_random.uniform(padding, self.HEIGHT - padding)
            )
            self.enemies.append({
                "pos": pygame.Vector2(orbit_center),
                "health": 10,
                "orbit_center": orbit_center,
                "orbit_radius": self.np_random.uniform(30, 80),
                "orbit_angle": self.np_random.uniform(0, 2 * math.pi),
                "orbit_speed": self.np_random.uniform(0.01, 0.03) * self.np_random.choice([-1, 1]),
                "fire_cooldown": self.np_random.integers(30, 90)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.02  # Small penalty per step to encourage speed

        # --- Handle Cooldowns ---
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
        for enemy in self.enemies:
            if enemy["fire_cooldown"] > 0:
                enemy["fire_cooldown"] -= 1

        # --- Handle Actions ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Player Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1

        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.last_move_direction = move_vec.copy()
        
        self.player_pos += move_vec * 4 # Player speed
        self.player_pos.x = np.clip(self.player_pos.x, 10, self.WIDTH - 10)
        self.player_pos.y = np.clip(self.player_pos.y, 10, self.HEIGHT - 10)

        # Player Firing
        if space_held and self.player_fire_cooldown == 0:
            self.player_fire_cooldown = 8 # Cooldown in frames
            proj_start_pos = self.player_pos + self.last_move_direction * 15
            self.player_projectiles.append({
                "pos": proj_start_pos,
                "vel": self.last_move_direction * 10 # Projectile speed
            })
            # Sound placeholder: pew!

        # --- Update Game Logic ---
        # Update enemies
        for enemy in self.enemies:
            enemy["orbit_angle"] += enemy["orbit_speed"]
            enemy["pos"].x = enemy["orbit_center"].x + math.cos(enemy["orbit_angle"]) * enemy["orbit_radius"]
            enemy["pos"].y = enemy["orbit_center"].y + math.sin(enemy["orbit_angle"]) * enemy["orbit_radius"]

            # Enemy firing
            if enemy["fire_cooldown"] <= 0:
                enemy["fire_cooldown"] = self.np_random.integers(100, 200)
                direction = (self.player_pos - enemy["pos"]).normalize()
                self.enemy_projectiles.append({
                    "pos": enemy["pos"].copy(),
                    "vel": direction * 5 # Enemy projectile speed
                })
                # Sound placeholder: zap!

        # Update projectiles
        self.player_projectiles = [p for p in self.player_projectiles if self._update_projectile(p)]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if self._update_projectile(p)]

        # --- Handle Collisions ---
        # Player projectiles vs Enemies
        remaining_enemies = []
        for enemy in self.enemies:
            is_hit = False
            for proj in self.player_projectiles:
                if (proj["pos"] - enemy["pos"]).length() < 12: # Collision radius
                    enemy["health"] -= 10
                    reward += 2
                    self.player_projectiles.remove(proj)
                    is_hit = True
                    break
            
            if enemy["health"] <= 0:
                self.score += 10
                reward += 10
                self.explosions.append({"pos": enemy["pos"].copy(), "radius": 5, "max_radius": 30, "life": 1})
                # Sound placeholder: boom!
            else:
                remaining_enemies.append(enemy)
        self.enemies = remaining_enemies
        
        # Enemy projectiles vs Player
        for proj in self.enemy_projectiles:
            if (proj["pos"] - self.player_pos).length() < 12:
                self.player_health -= 10
                reward -= 5
                self.enemy_projectiles.remove(proj)
                self.explosions.append({"pos": self.player_pos.copy(), "radius": 3, "max_radius": 20, "life": 0.5})
                # Sound placeholder: player hit!
                break

        # Update explosions
        self.explosions = [e for e in self.explosions if self._update_explosion(e)]

        # --- Termination ---
        self.steps += 1
        terminated = False
        if self.player_health <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
        elif not self.enemies:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_projectile(self, p):
        p["pos"] += p["vel"]
        return 0 < p["pos"].x < self.WIDTH and 0 < p["pos"].y < self.HEIGHT

    def _update_explosion(self, e):
        e["life"] -= 0.05
        e["radius"] = (1 - e["life"]) * e["max_radius"]
        return e["life"] > 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_ARENA_BORDER, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Render explosions
        for e in self.explosions:
            alpha = int(255 * e["life"])
            if alpha > 0:
                color = (*self.COLOR_EXPLOSION, alpha)
                temp_surf = pygame.Surface((e["radius"]*2, e["radius"]*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (e["radius"], e["radius"]), e["radius"])
                self.screen.blit(temp_surf, (int(e["pos"].x - e["radius"]), int(e["pos"].y - e["radius"])), special_flags=pygame.BLEND_RGBA_ADD)

        # Render enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"].x), int(enemy["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, *pos, 10, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, *pos, 10, self.COLOR_ENEMY_GLOW)

        # Render player
        player_rect = pygame.Rect(0, 0, 16, 16)
        player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Player glow
        glow_radius = 20
        temp_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*self.COLOR_PLAYER_GLOW, 80), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (player_rect.centerx - glow_radius, player_rect.centery - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, player_rect, 1)


        # Render projectiles
        for p in self.player_projectiles:
            start = p["pos"] - p["vel"].normalize() * 5
            end = p["pos"] + p["vel"].normalize() * 5
            pygame.draw.aaline(self.screen, self.COLOR_PLAYER_PROJ, (start.x, start.y), (end.x, end.y), 2)
        for p in self.enemy_projectiles:
            start = p["pos"] - p["vel"].normalize() * 4
            end = p["pos"] + p["vel"].normalize() * 4
            pygame.draw.aaline(self.screen, self.COLOR_ENEMY_PROJ, (start.x, start.y), (end.x, end.y), 2)

        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.player_health / 100.0)
        bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_pct), 20))
        health_text = self.font_small.render("HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
        
        # Enemy Count
        enemy_text = self.font_small.render(f"ENEMIES: {len(self.enemies)}", True, self.COLOR_UI_TEXT)
        enemy_rect = enemy_text.get_rect(topright=(self.WIDTH - 10, 35))
        self.screen.blit(enemy_text, enemy_rect)

        # Game Over / Win Message
        if self.game_over:
            if not self.enemies:
                message = "YOU WIN!"
            else:
                message = "GAME OVER"
            
            end_text = self.font_large.render(message, True, self.COLOR_UI_TEXT)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "enemies_remaining": len(self.enemies)
        }
    
    def render(self):
        # This method is not strictly required by the prompt, but is useful for human play.
        # It returns the same observation array.
        return self._get_observation()
        
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

# Example usage for human play
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Arcade Annihilator")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Action mapping for human control
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over. Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()