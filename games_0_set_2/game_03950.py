
# Generated: 2025-08-28T00:56:20.307597
# Source Brief: brief_03950.md
# Brief Index: 3950

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import heapq
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor, Space to place a block. Press Shift to start the wave."
    )

    game_description = (
        "Defend your fortress core against waves of enemies by strategically placing blocks to alter their path."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 15
    CELL_SIZE = 20
    GRID_AREA_WIDTH = GRID_WIDTH * CELL_SIZE
    GRID_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE
    GRID_OFFSET_X = SCREEN_WIDTH - GRID_AREA_WIDTH - 20
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_AREA_HEIGHT) // 2

    MAX_STEPS = 5000
    MAX_WAVES = 10
    INITIAL_BLOCKS = 50
    BLOCK_HEALTH = 3

    # --- Colors ---
    COLOR_BG = (44, 62, 80) # #2c3e50
    COLOR_GRID = (52, 73, 94) # #34495e
    COLOR_CORE = (52, 152, 219) # #3498db
    COLOR_CORE_GLOW = (52, 152, 219, 50)
    COLOR_BLOCK_3 = (46, 204, 113) # #2ecc71
    COLOR_BLOCK_2 = (241, 196, 15) # #f1c40f
    COLOR_BLOCK_1 = (230, 126, 34) # #e67e22
    COLOR_ENEMY = (231, 76, 60) # #e74c3c
    COLOR_CURSOR = (241, 196, 15, 150) # Transparent yellow
    COLOR_TEXT = (236, 240, 241) # #ecf0f1
    COLOR_INFO_BG = (35, 47, 62)

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

        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.blocks = {}
        self.enemies = []
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.game_phase = 'placement' # 'placement' or 'wave'
        
        self.wave_number = 1
        self.core_pos = (self.GRID_WIDTH - 1, self.GRID_HEIGHT // 2)
        self.core_health = 1

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.grid[self.core_pos] = 2 # 2 for core
        
        self.blocks = {}
        self.blocks_remaining = self.INITIAL_BLOCKS
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self.enemies = []
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        step_reward = 0
        terminated = False
        
        # --- Handle Input and Phase Transitions ---
        if self.game_phase == 'placement':
            self._handle_placement_phase(movement, space_held, shift_held)
        elif self.game_phase == 'wave':
            wave_reward, wave_ended = self._handle_wave_phase()
            step_reward += wave_reward
            if wave_ended:
                self.game_phase = 'placement'
                self.wave_number += 1
                step_reward += 1.0 # Survive wave reward
                # sfx: wave_complete_chime

                if self.wave_number > self.MAX_WAVES:
                    self.win = True
                    self.game_over = True

        # --- Update Game State ---
        self._update_particles()
        
        if self.core_health <= 0:
            self.game_over = True
            
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        # --- Final Rewards on Termination ---
        if self.game_over and not terminated:
            if self.win:
                step_reward += 50.0 # Win game reward
            elif self.core_health <= 0:
                step_reward -= 50.0 # Lose game penalty

        terminated = self.game_over
        self.score += step_reward
        
        # Tick clock for auto-advance
        self.clock.tick(30)

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_placement_phase(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Place block
        if space_held and not self.prev_space_held:
            cursor_tuple = tuple(self.cursor_pos)
            if self.grid[cursor_tuple] == 0 and self.blocks_remaining > 0:
                self.grid[cursor_tuple] = 1 # 1 for block
                self.blocks[cursor_tuple] = self.BLOCK_HEALTH
                self.blocks_remaining -= 1
                # sfx: block_place.wav
        
        # Start wave
        if shift_held and not self.prev_shift_held:
            self.game_phase = 'wave'
            self._spawn_enemies()
            for enemy in self.enemies:
                enemy['path'] = self._find_path(tuple(map(int, enemy['pos'])), self.core_pos)
            # sfx: wave_start.wav

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _handle_wave_phase(self):
        reward = 0
        if not self.enemies:
            return reward, True # Wave ended

        enemies_to_remove = []
        recalculate_paths = False
        
        for i, enemy in enumerate(self.enemies):
            if not enemy['path']: # No path to core
                continue

            # --- Movement ---
            target_node = enemy['path'][0]
            target_pos = [ (target_node[0] + 0.5) * self.CELL_SIZE, (target_node[1] + 0.5) * self.CELL_SIZE ]
            
            direction = [target_pos[0] - enemy['pixel_pos'][0], target_pos[1] - enemy['pixel_pos'][1]]
            dist = math.hypot(*direction)
            
            if dist < enemy['speed']:
                enemy['pixel_pos'] = target_pos
                enemy['path'].pop(0)
                enemy['pos'] = target_node
            else:
                direction_norm = [d / dist for d in direction]
                enemy['pixel_pos'][0] += direction_norm[0] * enemy['speed']
                enemy['pixel_pos'][1] += direction_norm[1] * enemy['speed']

            # --- Collision ---
            grid_pos = (int(enemy['pixel_pos'][0] / self.CELL_SIZE), int(enemy['pixel_pos'][1] / self.CELL_SIZE))
            
            if grid_pos in self.blocks:
                self.blocks[grid_pos] -= 1
                self._create_particles(enemy['pixel_pos'], 10, self.COLOR_BLOCK_3)
                # sfx: block_hit.wav
                if self.blocks[grid_pos] <= 0:
                    del self.blocks[grid_pos]
                    self.grid[grid_pos] = 0
                    recalculate_paths = True
                    # sfx: block_destroy.wav
                enemies_to_remove.append(i)
                reward += 0.1 # Kill enemy reward
                # sfx: enemy_die.wav
            
            elif grid_pos == self.core_pos:
                self.core_health -= 1
                self._create_particles(enemy['pixel_pos'], 20, self.COLOR_CORE)
                # sfx: core_hit.wav
                enemies_to_remove.append(i)

        # Remove defeated enemies
        for i in sorted(enemies_to_remove, reverse=True):
            self._create_particles(self.enemies[i]['pixel_pos'], 20, self.COLOR_ENEMY)
            del self.enemies[i]
            
        # Recalculate paths if a block was destroyed
        if recalculate_paths:
            for enemy in self.enemies:
                enemy['pos'] = (int(enemy['pixel_pos'][0] / self.CELL_SIZE), int(enemy['pixel_pos'][1] / self.CELL_SIZE))
                enemy['path'] = self._find_path(enemy['pos'], self.core_pos)

        return reward, not self.enemies

    def _spawn_enemies(self):
        num_enemies = 3 + (self.wave_number - 1) * 2
        speed_multiplier = 1.0 + ((self.wave_number - 1) // 2) * 0.2
        base_speed = self.CELL_SIZE / 30.0 * 2.0 # 2 cells per second at 30fps

        for _ in range(num_enemies):
            spawn_y = self.np_random.integers(0, self.GRID_HEIGHT)
            pos = [0, spawn_y]
            self.enemies.append({
                'pos': pos,
                'pixel_pos': [(pos[0] + 0.5) * self.CELL_SIZE, (pos[1] + 0.5) * self.CELL_SIZE],
                'speed': base_speed * speed_multiplier,
                'path': []
            })
    
    def _find_path(self, start, end):
        open_set = [(0, start)] # (f_cost, pos)
        came_from = {}
        g_cost = { (x,y): float('inf') for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT) }
        g_cost[start] = 0
        
        f_cost = { (x,y): float('inf') for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT) }
        f_cost[start] = abs(start[0] - end[0]) + abs(start[1] - end[1])

        open_set_hash = {start}

        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.GRID_WIDTH and 0 <= neighbor[1] < self.GRID_HEIGHT):
                    continue
                if self.grid[neighbor] == 1: # Obstacle
                    continue

                tentative_g_cost = g_cost[current] + 1
                if tentative_g_cost < g_cost[neighbor]:
                    came_from[neighbor] = current
                    g_cost[neighbor] = tentative_g_cost
                    h = abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                    f_cost[neighbor] = tentative_g_cost + h
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_cost[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        return [] # No path found

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            start = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_AREA_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_AREA_WIDTH, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Draw core
        cx, cy = self.core_pos
        core_rect = pygame.Rect(self.GRID_OFFSET_X + cx * self.CELL_SIZE, self.GRID_OFFSET_Y + cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.gfxdraw.filled_circle(self.screen, core_rect.centerx, core_rect.centery, int(self.CELL_SIZE * 1.5), self.COLOR_CORE_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_CORE, core_rect)

        # Draw blocks
        for pos, health in self.blocks.items():
            color = self.COLOR_BLOCK_1 if health == 1 else self.COLOR_BLOCK_2 if health == 2 else self.COLOR_BLOCK_3
            rect = pygame.Rect(self.GRID_OFFSET_X + pos[0] * self.CELL_SIZE, self.GRID_OFFSET_Y + pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, color, rect.inflate(-2, -2))
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in color), rect, 2)

        # Draw enemies
        for enemy in self.enemies:
            px, py = enemy['pixel_pos']
            pos = (self.GRID_OFFSET_X + int(px), self.GRID_OFFSET_Y + int(py))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.CELL_SIZE // 3, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.CELL_SIZE // 3, self.COLOR_ENEMY)

        # Draw cursor
        if self.game_phase == 'placement':
            cursor_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            cursor_surf.fill(self.COLOR_CURSOR)
            self.screen.blit(cursor_surf, (self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE, self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE))

        # Draw particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color_with_alpha = p['color'] + (alpha,) if len(p['color']) == 3 else (p['color'][0], p['color'][1], p['color'][2], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), color_with_alpha)

    def _render_ui(self):
        # UI Background
        ui_width = self.GRID_OFFSET_X - 20
        pygame.draw.rect(self.screen, self.COLOR_INFO_BG, (0, 0, ui_width, self.SCREEN_HEIGHT))

        # Score
        score_text = self.font_large.render(f"{int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (ui_width / 2 - score_text.get_width() / 2, 20))

        # Wave
        wave_label = self.font_medium.render("WAVE", True, self.COLOR_TEXT)
        self.screen.blit(wave_label, (20, 70))
        wave_val = self.font_large.render(f"{self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_val, (20, 95))

        # Blocks
        block_label = self.font_medium.render("BLOCKS", True, self.COLOR_TEXT)
        self.screen.blit(block_label, (20, 150))
        block_val = self.font_large.render(f"{self.blocks_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(block_val, (20, 175))

        # Phase Info
        if self.game_phase == 'placement' and not self.game_over:
            prompt_text = self.font_small.render("Press SHIFT to start wave", True, self.COLOR_TEXT)
            self.screen.blit(prompt_text, (20, self.SCREEN_HEIGHT - 40))
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            msg_render = self.font_large.render(message, True, self.COLOR_TEXT)
            msg_rect = msg_render.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_render, msg_rect)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _create_particles(self, pos, count, color):
        grid_pos = [pos[0] + self.GRID_OFFSET_X, pos[1] + self.GRID_OFFSET_Y]
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.5)
            life = random.randint(10, 25)
            self.particles.append({
                'pos': list(grid_pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'size': random.uniform(1, 3),
                'color': color
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "blocks_remaining": self.blocks_remaining,
            "game_phase": self.game_phase,
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
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    
    # --- Manual Control Mapping ---
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Create a display for human playing
    pygame.display.set_caption("Block Fortress")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    while not done:
        # --- Action gathering from keyboard ---
        movement_action = 0 # no-op
        space_action = 0
        shift_action = 0

        keys = pygame.key.get_pressed()
        for key, move_id in key_map.items():
            if keys[key]:
                movement_action = move_id
                break
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Rendering for human ---
        # The observation is already a rendered frame, we just need to display it.
        # Pygame's coordinate system (0,0 at top-left) is different from np.transpose's result.
        # We need to flip it back.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling (for closing the window) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Game Over! Final Score: {total_reward}")
    env.close()