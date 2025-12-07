
# Generated: 2025-08-28T01:13:31.562474
# Source Brief: brief_04042.md
# Brief Index: 4042

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move the placement cursor. "
        "Press space to place the selected tower. Each space press also cycles to the next tower type."
    )

    game_description = (
        "A minimalist tower defense game. Place towers to defend your base from waves of enemies. Survive for 3000 steps to win."
    )

    auto_advance = False

    # --- Constants ---
    # Game parameters
    MAX_STEPS = 3000
    GRID_WIDTH, GRID_HEIGHT = 20, 12
    CELL_SIZE = 30
    PLAY_AREA_WIDTH = GRID_WIDTH * CELL_SIZE
    PLAY_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLAY_AREA_X_OFFSET = (SCREEN_WIDTH - PLAY_AREA_WIDTH) // 2  # 20
    PLAY_AREA_Y_OFFSET = (SCREEN_HEIGHT - PLAY_AREA_HEIGHT) // 2 # 20

    # Colors
    COLOR_BG = (44, 62, 80)  # Dark blue-grey
    COLOR_GRID = (52, 73, 94) # Slightly lighter blue-grey
    COLOR_BASE = (46, 204, 113) # Green
    COLOR_ENEMY = (231, 76, 60) # Red
    COLOR_PROJECTILE = (236, 240, 241) # White
    COLOR_CURSOR = (26, 188, 156, 128) # Transparent turquoise
    COLOR_TEXT = (236, 240, 241) # White

    TOWER_SPECS = {
        1: {"color": (52, 152, 219), "cost": 15, "range": 3.5 * CELL_SIZE, "cooldown": 30, "projectile_speed": 5, "damage": 10}, # Blue
        2: {"color": (241, 196, 15), "cost": 25, "range": 4.5 * CELL_SIZE, "cooldown": 15, "projectile_speed": 8, "damage": 5}, # Yellow
    }

    # Initial state
    INITIAL_BASE_HEALTH = 100
    INITIAL_MONEY = 50
    INITIAL_ENEMY_SPAWN_RATE = 0.01

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_health = pygame.font.SysFont("monospace", 12, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.tower_grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # This will be initialized in reset()
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.money = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 1
        self.enemy_spawn_rate = 0.0
        self.last_space_held = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.INITIAL_BASE_HEALTH
        self.money = self.INITIAL_MONEY
        self.enemy_spawn_rate = self.INITIAL_ENEMY_SPAWN_RATE
        
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.tower_grid.fill(0)
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 1
        self.last_space_held = False
        
        self.base_pos_grid = (self.GRID_WIDTH - 1, self.GRID_HEIGHT // 2 - 1)
        self.base_rect = pygame.Rect(
            self.PLAY_AREA_X_OFFSET + self.base_pos_grid[0] * self.CELL_SIZE,
            self.PLAY_AREA_Y_OFFSET + self.base_pos_grid[1] * self.CELL_SIZE,
            self.CELL_SIZE,
            2 * self.CELL_SIZE
        )
        self.base_center = self.base_rect.center
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # 1. Handle Player Input
        self._handle_input(movement, space_held)

        # 2. Update Game Logic
        reward += self._update_towers()
        self._update_projectiles()
        enemy_updates = self._update_enemies()
        reward += enemy_updates["reward"]
        self.base_health -= enemy_updates["base_damage"]
        self._update_particles()
        
        # 3. Spawn new enemies
        self._spawn_enemies()

        # 4. Update step counter and difficulty
        self.steps += 1
        if self.steps % 100 == 0:
            self.enemy_spawn_rate += 0.001

        # 5. Check for termination
        terminated = False
        if self.base_health <= 0:
            self.base_health = 0
            reward -= 100
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            reward += 100
            self.game_over = True
            terminated = True
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Place tower on space press (rising edge)
        is_space_press = space_held and not self.last_space_held
        if is_space_press:
            tower_spec = self.TOWER_SPECS[self.selected_tower_type]
            cx, cy = self.cursor_pos
            is_base_area = cx >= self.base_pos_grid[0] and cy >= self.base_pos_grid[1] and cy < self.base_pos_grid[1] + 2

            if self.money >= tower_spec["cost"] and self.tower_grid[cx, cy] == 0 and not is_base_area:
                self.money -= tower_spec["cost"]
                self.tower_grid[cx, cy] = self.selected_tower_type
                
                screen_pos = (
                    self.PLAY_AREA_X_OFFSET + (cx + 0.5) * self.CELL_SIZE,
                    self.PLAY_AREA_Y_OFFSET + (cy + 0.5) * self.CELL_SIZE
                )
                
                new_tower = {
                    "type": self.selected_tower_type,
                    "pos": screen_pos,
                    "cooldown": 0,
                    "spec": tower_spec
                }
                self.towers.append(new_tower)
                # sfx: tower_place.wav
                self._create_particles(screen_pos, tower_spec["color"], 20, 5, 20) # Placement effect

            # Cycle tower type for next placement
            self.selected_tower_type = 2 if self.selected_tower_type == 1 else 1

        self.last_space_held = space_held
    
    def _spawn_enemies(self):
        if self.np_random.random() < self.enemy_spawn_rate:
            spawn_y = self.PLAY_AREA_Y_OFFSET + (self.np_random.random() * 0.8 + 0.1) * self.PLAY_AREA_HEIGHT
            spawn_x = self.PLAY_AREA_X_OFFSET
            
            self.enemies.append({
                "pos": np.array([spawn_x, spawn_y], dtype=float),
                "health": 20,
                "max_health": 20,
                "speed": 1.0 + self.steps / 2000.0, # Speed increases over time
                "id": self.np_random.integers(1, 1_000_000)
            })

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] == 0:
                target = None
                min_dist = tower["spec"]["range"] ** 2
                
                for enemy in self.enemies:
                    dist_sq = (enemy["pos"][0] - tower["pos"][0])**2 + (enemy["pos"][1] - tower["pos"][1])**2
                    if dist_sq < min_dist:
                        min_dist = dist_sq
                        target = enemy
                
                if target:
                    # sfx: shoot.wav
                    tower["cooldown"] = tower["spec"]["cooldown"]
                    self.projectiles.append({
                        "pos": np.array(tower["pos"], dtype=float),
                        "target_id": target["id"],
                        "target_pos": np.copy(target["pos"]), # Snapshot of position
                        "spec": tower["spec"]
                    })
        return reward

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            target = next((e for e in self.enemies if e["id"] == proj["target_id"]), None)
            
            if target:
                proj["target_pos"] = target["pos"] # Update target position for homing
            
            direction = proj["target_pos"] - proj["pos"]
            dist = np.linalg.norm(direction)

            if dist < proj["spec"]["projectile_speed"]:
                # Hit
                if target:
                    target["health"] -= proj["spec"]["damage"]
                    self._create_particles(target["pos"], self.COLOR_PROJECTILE, 10, 2, 10) # Hit effect
                    # sfx: hit.wav
                self.projectiles.remove(proj)
            else:
                # Move
                direction /= dist
                proj["pos"] += direction * proj["spec"]["projectile_speed"]

                # Remove if out of bounds
                if not (0 < proj["pos"][0] < self.SCREEN_WIDTH and 0 < proj["pos"][1] < self.SCREEN_HEIGHT):
                    self.projectiles.remove(proj)

    def _update_enemies(self):
        reward = 0
        base_damage = 0
        for enemy in self.enemies[:]:
            # Check for death
            if enemy["health"] <= 0:
                reward += 1
                self.money += 5
                self._create_particles(enemy["pos"], self.COLOR_ENEMY, 30, 4, 30) # Death effect
                # sfx: enemy_destroy.wav
                self.enemies.remove(enemy)
                continue

            # Move towards base
            direction = self.base_center - enemy["pos"]
            dist = np.linalg.norm(direction)

            if dist < enemy["speed"]:
                # Reached base
                base_damage += 10
                self._create_particles(enemy["pos"], self.COLOR_BASE, 20, 5, 25, (1,0)) # Base hit effect
                # sfx: base_damage.wav
                self.enemies.remove(enemy)
            else:
                direction /= dist
                enemy["pos"] += direction * enemy["speed"]
        
        return {"reward": reward, "base_damage": base_damage}

    def _create_particles(self, pos, color, count, max_speed, max_life, force_dir=None):
        for _ in range(count):
            if force_dir:
                angle = math.atan2(force_dir[1], force_dir[0]) + (self.np_random.random() - 0.5) * math.pi/2
            else:
                angle = self.np_random.random() * 2 * math.pi
            
            speed = self.np_random.random() * max_speed
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(10, max_life)
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": life,
                "max_life": life,
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0] / 5.0
            p["pos"][1] += p["vel"][1] / 5.0
            p["vel"][0] *= 0.9
            p["vel"][1] *= 0.9
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            start = (self.PLAY_AREA_X_OFFSET + x * self.CELL_SIZE, self.PLAY_AREA_Y_OFFSET)
            end = (self.PLAY_AREA_X_OFFSET + x * self.CELL_SIZE, self.PLAY_AREA_Y_OFFSET + self.PLAY_AREA_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.PLAY_AREA_X_OFFSET, self.PLAY_AREA_Y_OFFSET + y * self.CELL_SIZE)
            end = (self.PLAY_AREA_X_OFFSET + self.PLAY_AREA_WIDTH, self.PLAY_AREA_Y_OFFSET + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Draw base
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.base_rect)
        
        # Draw towers
        for tower in self.towers:
            pygame.gfxdraw.filled_circle(self.screen, int(tower["pos"][0]), int(tower["pos"][1]), self.CELL_SIZE // 3, tower["spec"]["color"])
            pygame.gfxdraw.aacircle(self.screen, int(tower["pos"][0]), int(tower["pos"][1]), self.CELL_SIZE // 3, tower["spec"]["color"])

        # Draw enemies
        for enemy in self.enemies:
            pos_int = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 8, self.COLOR_ENEMY)
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            bar_width = 16
            bar_x = pos_int[0] - bar_width // 2
            bar_y = pos_int[1] - 14
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (bar_x, bar_y, bar_width * health_ratio, 3))

        # Draw projectiles
        for proj in self.projectiles:
            pos_int = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, (pos_int[0]-2, pos_int[1]-2, 4, 4))
        
        # Draw particles
        for p in self.particles:
            life_ratio = p["life"] / p["max_life"]
            size = int(life_ratio * 5)
            if size > 0:
                color = tuple(c * life_ratio for c in p["color"])
                pygame.draw.circle(self.screen, color, (int(p["pos"][0]), int(p["pos"][1])), size)

        # Draw cursor
        cursor_color = self.TOWER_SPECS[self.selected_tower_type]["color"]
        cursor_rect = pygame.Rect(
            self.PLAY_AREA_X_OFFSET + self.cursor_pos[0] * self.CELL_SIZE,
            self.PLAY_AREA_Y_OFFSET + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill((*cursor_color, 80))
        self.screen.blit(s, cursor_rect.topleft)
        pygame.draw.rect(self.screen, cursor_color, cursor_rect, 2)


    def _render_ui(self):
        # Base Health Bar
        health_ratio = self.base_health / self.INITIAL_BASE_HEALTH
        health_bar_width = 100
        health_bar_rect = pygame.Rect(self.base_rect.left - health_bar_width - 10, self.base_rect.centery - 10, health_bar_width, 20)
        pygame.draw.rect(self.screen, self.COLOR_GRID, health_bar_rect, 2, 3)
        fill_rect = pygame.Rect(health_bar_rect.left + 2, health_bar_rect.top + 2, (health_bar_width - 4) * health_ratio, 16)
        pygame.draw.rect(self.screen, self.COLOR_BASE, fill_rect, 0, 3)
        health_text = self.font_health.render(f"{int(self.base_health)}/{self.INITIAL_BASE_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (health_bar_rect.centerx - health_text.get_width() // 2, health_bar_rect.centery - health_text.get_height() // 2))

        # UI Text
        texts = [
            f"TIME: {self.steps}/{self.MAX_STEPS}",
            f"SCORE: {int(self.score)}",
            f"MONEY: ${self.money}",
        ]
        for i, text in enumerate(texts):
            surf = self.font_ui.render(text, True, self.COLOR_TEXT)
            self.screen.blit(surf, (10, 10 + i * 20))

        # Selected Tower UI
        st_text = self.font_ui.render("NEXT TOWER:", True, self.COLOR_TEXT)
        self.screen.blit(st_text, (self.SCREEN_WIDTH - 150, 10))
        tower_spec = self.TOWER_SPECS[self.selected_tower_type]
        pygame.gfxdraw.filled_circle(self.screen, self.SCREEN_WIDTH - 100, 45, self.CELL_SIZE // 3, tower_spec["color"])
        pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH - 100, 45, self.CELL_SIZE // 3, tower_spec["color"])
        cost_text = self.font_ui.render(f"COST: ${tower_spec['cost']}", True, tower_spec["color"])
        self.screen.blit(cost_text, (self.SCREEN_WIDTH - 150, 60))


    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))
        
        if self.base_health <= 0:
            text = "BASE DESTROYED"
            color = self.COLOR_ENEMY
        else:
            text = "VICTORY"
            color = self.COLOR_BASE
            
        text_surf = self.font_game_over.render(text, True, color)
        pos = (self.SCREEN_WIDTH // 2 - text_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - text_surf.get_height() // 2)
        self.screen.blit(text_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "money": self.money,
            "enemies": len(self.enemies),
            "towers": len(self.towers),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
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
    # It will not be executed when imported by a training script.
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Must be set for pygame.display to work with a dummy driver
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # To run headlessly and see the output, we need a display.
    # For actual training, this part would be removed.
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    # Game loop for human play
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
            # Wait for reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False
        
        # Since auto_advance is False, we need to control the speed for human play
        env.clock.tick(30) # Limit to 30 steps per second for playability
        
    env.close()