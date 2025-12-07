import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


# Set headless mode for Pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor, Space to place selected block, Shift to cycle block type. Press nothing to let enemies advance."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your fortress core against waves of enemies by strategically placing defensive blocks on the grid."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 12
        self.CELL_SIZE = 32
        self.UI_HEIGHT = self.HEIGHT - (self.GRID_HEIGHT * self.CELL_SIZE) # Should be 16px, but calculation is safer

        self.MAX_STEPS = 1000
        self.MAX_WAVES = 10

        # Colors
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_GRID = (45, 45, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CORE = (255, 220, 0)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_PROJECTILE = (100, 200, 255)
        self.BLOCK_COLORS = {
            "WALL": (50, 200, 50),
            "TURRET": (50, 150, 255)
        }
        self.COLOR_TEXT = (230, 230, 230)

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
        try:
            self.font_small = pygame.font.SysFont("Consolas", 14)
            self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 18)
            self.font_large = pygame.font.Font(None, 60)

        # State variables
        self.grid = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.core_pos = [0, 0]
        self.core_health = 0
        self.max_core_health = 100
        self.wave = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.block_types = ["WALL", "TURRET"]
        self.selected_block_idx = 0
        self.block_inventory = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.wave = 1

        self.grid = [[{"type": "EMPTY", "health": 0, "max_health": 0} for _ in range(self.GRID_HEIGHT)] for _ in range(self.GRID_WIDTH)]
        self.core_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.grid[self.core_pos[0]][self.core_pos[1]] = {"type": "CORE", "health": 100, "max_health": 100}
        self.core_health = 100

        self.cursor_pos = [self.core_pos[0] - 2, self.core_pos[1]]

        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.selected_block_idx = 0
        self._replenish_blocks()
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        shift_pressed = action[2] == 1  # Boolean

        # Update game logic
        reward = 0
        self.steps += 1

        # 1. Player Action Phase
        self._handle_player_action(movement, space_pressed, shift_pressed)

        # 2. World Update Phase
        step_reward = self._update_world()
        reward += step_reward

        # 3. State Check Phase
        if self.core_health <= 0:
            self.game_over = True
            self.victory = False
            reward -= 100
            self._add_explosion(self.core_pos, self.COLOR_CORE, 50)

        if not self.enemies and not self.game_over:
            reward += 1
            self.wave += 1
            if self.wave > self.MAX_WAVES:
                self.game_over = True
                self.victory = True
                reward += 100
            else:
                self._spawn_wave()
                self._replenish_blocks()

        self.score += reward
        terminated = self.game_over or self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_player_action(self, movement, space_pressed, shift_pressed):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Cycle block type
        if shift_pressed:
            self.selected_block_idx = (self.selected_block_idx + 1) % len(self.block_types)

        # Place block
        if space_pressed:
            x, y = self.cursor_pos
            selected_type = self.block_types[self.selected_block_idx]
            if self.grid[x][y]["type"] == "EMPTY" and self.block_inventory[selected_type] > 0:
                self.block_inventory[selected_type] -= 1

                block_stats = {"WALL": (50, 50), "TURRET": (25, 25)} # health, max_health
                health, max_health = block_stats[selected_type]

                self.grid[x][y] = {"type": selected_type, "health": health, "max_health": max_health}
                self._add_particle(self.cursor_pos, self.BLOCK_COLORS[selected_type], 10, 10, 'expand')

    def _update_world(self):
        reward = 0

        # Update particles
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['radius'] *= 0.95
            p['lifetime'] -= 1

        # Update turrets (fire projectiles)
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x][y]["type"] == "TURRET":
                    if self.np_random.random() < 0.1: # Firing rate
                        target = self._find_nearest_enemy([x, y], 5)
                        if target:
                            self.projectiles.append({"pos": [x, y], "target": target, "speed": 0.5})
                            self._add_particle([x,y], self.COLOR_PROJECTILE, 5, 5, 'static')

        # Update projectiles
        new_projectiles = []
        for proj in self.projectiles:
            target_pos = proj['target']['pos']
            proj_pixel_pos = self._grid_to_pixel(proj['pos'], as_int=False)
            target_pixel_pos = self._grid_to_pixel(target_pos, as_int=False)

            angle = math.atan2(target_pixel_pos[1] - proj_pixel_pos[1], target_pixel_pos[0] - proj_pixel_pos[0])
            proj['pos'][0] += math.cos(angle) * proj['speed']
            proj['pos'][1] += math.sin(angle) * proj['speed']

            if math.dist(proj['pos'], target_pos) < 0.5:
                proj['target']['health'] -= 10
                reward += 0.1
                self._add_particle(target_pos, self.COLOR_ENEMY, 8, 10, 'static')
            else:
                new_projectiles.append(proj)
        self.projectiles = new_projectiles

        # Update enemies
        enemies_to_remove = []
        for enemy in self.enemies:
            if enemy['health'] <= 0:
                enemies_to_remove.append(enemy)
                continue

            # Pathfinding and movement
            path = self._find_path(enemy['pos'], self.core_pos)
            
            attacked = False
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                check_pos = [enemy['pos'][0] + dx, enemy['pos'][1] + dy]
                if 0 <= check_pos[0] < self.GRID_WIDTH and 0 <= check_pos[1] < self.GRID_HEIGHT:
                    block = self.grid[check_pos[0]][check_pos[1]]
                    if block['type'] != "EMPTY":
                        block['health'] -= enemy['damage']
                        self._add_particle(check_pos, (255,100,100), 5, 5, 'static')
                        if block['health'] <= 0:
                            if block['type'] == "CORE":
                                self.core_health = 0
                            else:
                                block_type_before_destruction = block['type']
                                self.grid[check_pos[0]][check_pos[1]] = {"type": "EMPTY", "health": 0, "max_health": 0}
                                reward -= 0.01
                                self._add_explosion(check_pos, self.BLOCK_COLORS.get(block_type_before_destruction, (100,100,100)), 20)
                        attacked = True
                        break
            if not attacked and path and len(path) > 1:
                next_pos = path[1]
                enemy['pos'] = next_pos
        
        # Cleanup dead enemies
        self.enemies = [e for e in self.enemies if e['health'] > 0]
        self.core_health = self.grid[self.core_pos[0]][self.core_pos[1]]['health']

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.UI_HEIGHT), (px, self.HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE + self.UI_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.WIDTH, py))

        # Draw blocks and core
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                block = self.grid[x][y]
                if block['type'] != "EMPTY":
                    color = self.COLOR_CORE if block['type'] == "CORE" else self.BLOCK_COLORS[block['type']]
                    px, py = self._grid_to_pixel([x, y])
                    rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
                    pygame.draw.rect(self.screen, color, rect.inflate(-4, -4))
                    if block['max_health'] > 0:
                        health_pct = max(0, block['health'] / block['max_health'])
                        bar_color = (0, 255, 0) if health_pct > 0.5 else ((255, 255, 0) if health_pct > 0.2 else (255, 0, 0))
                        pygame.draw.rect(self.screen, (50,50,50), (px, py + self.CELL_SIZE - 4, self.CELL_SIZE, 4))
                        pygame.draw.rect(self.screen, bar_color, (px, py + self.CELL_SIZE - 4, int(self.CELL_SIZE * health_pct), 4))

        # Draw enemies
        for enemy in self.enemies:
            px, py = self._grid_to_pixel(enemy['pos'])
            center = (px + self.CELL_SIZE // 2, py + self.CELL_SIZE // 2)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.CELL_SIZE // 3, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.CELL_SIZE // 3, self.COLOR_ENEMY)
            health_pct = max(0, enemy['health'] / enemy['max_health'])
            pygame.draw.rect(self.screen, (50,50,50), (px, py - 6, self.CELL_SIZE, 4))
            pygame.draw.rect(self.screen, (0,255,0), (px, py - 6, int(self.CELL_SIZE * health_pct), 4))

        # Draw projectiles
        for proj in self.projectiles:
            px, py = self._grid_to_pixel(proj['pos'], as_int=False)
            center = (int(px + self.CELL_SIZE // 2), int(py + self.CELL_SIZE // 2))
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], 3, self.COLOR_PROJECTILE)

        # Draw particles
        for p in self.particles:
            px, py = self._grid_to_pixel(p['pos'], as_int=False)
            center = (int(px + self.CELL_SIZE // 2), int(py + self.CELL_SIZE // 2))
            alpha = int(255 * (p['lifetime'] / p['start_life']))
            color_with_alpha = (*p['color'], alpha)
            try:
                if p['shape'] == 'circle':
                    pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], int(p['radius']), color_with_alpha)
                elif p['shape'] == 'expand':
                    pygame.gfxdraw.aacircle(self.screen, center[0], center[1], int(p['start_radius'] - p['radius']), color_with_alpha)
            except (ValueError, TypeError): # Handle potential color format issues
                pass


        # Draw cursor
        cx, cy = self._grid_to_pixel(self.cursor_pos)
        cursor_rect = pygame.Rect(cx, cy, self.CELL_SIZE, self.CELL_SIZE)
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill((*self.COLOR_CURSOR, 60))
        self.screen.blit(s, (cx, cy))
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)

    def _render_ui(self):
        pygame.draw.rect(self.screen, (15, 15, 25), (0, 0, self.WIDTH, self.UI_HEIGHT))
        self._draw_text(f"Wave: {self.wave}/{self.MAX_WAVES}", (10, 0), self.font_small)
        self._draw_text(f"Core HP: {int(self.core_health)}", (100, 0), self.font_small)

        selected_type = self.block_types[self.selected_block_idx]
        for i, b_type in enumerate(self.block_types):
            is_selected = (b_type == selected_type)
            color = self.COLOR_CURSOR if is_selected else self.BLOCK_COLORS[b_type]
            count = self.block_inventory[b_type]
            text_surf = self.font_small.render(f"[{b_type}: {count}]", True, color)
            x_pos = self.WIDTH - 250 + i * 100
            self.screen.blit(text_surf, (x_pos, 0))

        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            msg = "VICTORY" if self.victory else "DEFEAT"
            color = (100, 255, 100) if self.victory else (255, 100, 100)
            self._draw_text(msg, (self.WIDTH / 2, self.HEIGHT / 2 - 30), self.font_large, color, center=True)
            self._draw_text(f"Final Score: {int(self.score)}", (self.WIDTH / 2, self.HEIGHT / 2 + 30), self.font_small, self.COLOR_TEXT, center=True)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave, "core_health": self.core_health}

    def _grid_to_pixel(self, grid_pos, as_int=True):
        px = grid_pos[0] * self.CELL_SIZE
        py = grid_pos[1] * self.CELL_SIZE + self.UI_HEIGHT
        return (int(px), int(py)) if as_int else (px, py)

    def _draw_text(self, text, pos, font, color=None, center=False):
        if color is None: color = self.COLOR_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _spawn_wave(self):
        num_enemies = 2 + self.wave
        enemy_health = 8 + 2 * self.wave
        enemy_damage = 1 + self.wave

        for _ in range(num_enemies):
            while True:
                side = self.np_random.integers(4)
                if side == 0: pos = [0, self.np_random.integers(self.GRID_HEIGHT)] # Left
                elif side == 1: pos = [self.GRID_WIDTH-1, self.np_random.integers(self.GRID_HEIGHT)] # Right
                elif side == 2: pos = [self.np_random.integers(self.GRID_WIDTH), 0] # Top
                else: pos = [self.np_random.integers(self.GRID_WIDTH), self.GRID_HEIGHT-1] # Bottom

                if self.grid[pos[0]][pos[1]]['type'] == 'EMPTY' and tuple(pos) not in [tuple(e['pos']) for e in self.enemies]:
                    self.enemies.append({
                        "pos": pos,
                        "health": enemy_health,
                        "max_health": enemy_health,
                        "damage": enemy_damage,
                    })
                    break

    def _replenish_blocks(self):
        self.block_inventory = {"WALL": 5, "TURRET": 2}

    def _find_path(self, start, end):
        # Simple BFS pathfinding
        q = deque([[start]])
        visited = {tuple(start)}

        while q:
            path = q.popleft()
            pos = path[-1]

            if pos == end:
                return path

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = [pos[0] + dx, pos[1] + dy]
                if (0 <= next_pos[0] < self.GRID_WIDTH and
                    0 <= next_pos[1] < self.GRID_HEIGHT and
                    tuple(next_pos) not in visited and
                    self.grid[next_pos[0]][next_pos[1]]['type'] in ["EMPTY", "CORE"]):
                    
                    visited.add(tuple(next_pos))
                    new_path = path + [next_pos]
                    q.append(new_path)
        return None # No path

    def _find_nearest_enemy(self, pos, max_range):
        nearest = None
        min_dist = float('inf')
        for enemy in self.enemies:
            dist = math.dist(pos, enemy['pos'])
            if dist < min_dist and dist <= max_range:
                min_dist = dist
                nearest = enemy
        return nearest

    def _add_particle(self, grid_pos, color, radius, lifetime, p_type):
        if p_type == 'static':
             self.particles.append({
                'pos': list(grid_pos), 'vel': [0, 0], 'radius': radius, 'start_radius': radius,
                'color': color, 'lifetime': lifetime, 'start_life': lifetime, 'shape': 'circle'
            })
        elif p_type == 'expand':
             self.particles.append({
                'pos': list(grid_pos), 'vel': [0, 0], 'radius': 0, 'start_radius': radius,
                'color': color, 'lifetime': lifetime, 'start_life': lifetime, 'shape': 'expand'
            })

    def _add_explosion(self, grid_pos, color, count):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 0.2 + 0.05
            self.particles.append({
                'pos': list(grid_pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.integers(2, 5),
                'start_radius': 5,
                'color': color,
                'lifetime': self.np_random.integers(15, 30),
                'start_life': 30,
                'shape': 'circle'
            })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually.
    # It requires a windowed pygame environment.
    # To run, you might need to unset the SDL_VIDEODRIVER variable.
    # For example, in bash: `unset SDL_VIDEODRIVER && python your_script.py`
    
    # Unset the headless environment variable for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    try:
        env = GameEnv()
        # Re-initialize pygame with video display
        pygame.display.init()
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Block Fortress Defense")
        clock = pygame.time.Clock()
        obs, info = env.reset()
        running = True
        
        print(GameEnv.user_guide)

        while running:
            action = [0, 0, 0] # Default to no-op
            
            # Event handling
            events = pygame.event.get()
            made_move = False
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                # For turn-based, we can register an action on keydown
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    elif event.key == pygame.K_SPACE: action[1] = 1
                    elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                    elif event.key == pygame.K_RETURN: pass # No-op action [0,0,0] is a pass
                    else:
                        continue # Don't step if an unmapped key was pressed
                    
                    made_move = True

            if made_move or env.auto_advance:
                obs, reward, terminated, truncated, info = env.step(action)
                
                if reward != 0:
                    print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

                if terminated:
                    print("Game Over! Resetting in 3 seconds...")
                    surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    pygame.time.wait(3000)
                    obs, info = env.reset()

            # Render the current state
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            clock.tick(30) # Limit FPS

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()