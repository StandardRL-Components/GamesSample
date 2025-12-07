
# Generated: 2025-08-28T05:13:37.320134
# Source Brief: brief_02549.md
# Brief Index: 2549

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Space to attack in your last moved direction. "
        "Survive, collect gold, and find the exit."
    )

    game_description = (
        "A turn-based dungeon crawler. Navigate a procedurally generated maze, "
        "battle monsters, and collect gold to reach the exit and advance to the next level."
    )

    auto_advance = False

    # --- Constants ---
    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16  # Wider grid for better aspect ratio
    GRID_HEIGHT = 10
    TILE_SIZE = 40
    
    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_WALL = (60, 60, 80)
    COLOR_FLOOR = (40, 40, 55)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_PLAYER_OUTLINE = (255, 200, 200)
    COLOR_MONSTER = (150, 50, 255)
    COLOR_MONSTER_OUTLINE = (220, 180, 255)
    COLOR_GOLD = (255, 223, 0)
    COLOR_EXIT = (255, 215, 100)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HP_BAR = (50, 205, 50)
    COLOR_HP_BAR_BG = (120, 0, 0)

    # Game Parameters
    PLAYER_MAX_HP = 100
    MONSTER_BASE_HP = 20
    MONSTER_COUNT = 5
    GOLD_COUNT = 8
    GOLD_VALUE = 20
    MAX_STEPS = 1000

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
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # State variables are initialized in reset()
        self.grid = None
        self.player_pos = None
        self.player_hp = None
        self.last_move_dir = None
        self.monsters = None
        self.gold_piles = None
        self.exit_pos = None
        self.level = None
        self.gold = None
        self.steps = None
        self.game_over = None
        self.particles = None
        self.damage_flashes = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.level = 1
        self.gold = 0
        self.steps = 0
        self.game_over = False
        self.damage_flashes = collections.defaultdict(int)

        self._setup_level()
        
        return self._get_observation(), self._get_info()
    
    def _setup_level(self):
        """Generates a new dungeon level."""
        self.player_hp = self.PLAYER_MAX_HP
        self.last_move_dir = (1, 0) # Default attack direction
        self.particles = []

        # Generate dungeon layout
        self.grid, self.player_pos, self.exit_pos = self._generate_dungeon()
        
        # Get all valid floor tiles for spawning
        floor_tiles = []
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] == 0 and (x, y) != self.player_pos and (x, y) != self.exit_pos:
                    floor_tiles.append((x, y))
        
        self.np_random.shuffle(floor_tiles)

        # Place monsters
        self.monsters = []
        monster_hp = self.MONSTER_BASE_HP + (self.level - 1)
        for pos in floor_tiles[:self.MONSTER_COUNT]:
            self.monsters.append({
                "pos": list(pos), 
                "hp": monster_hp, 
                "max_hp": monster_hp
            })
        
        # Place gold
        self.gold_piles = floor_tiles[self.MONSTER_COUNT : self.MONSTER_COUNT + self.GOLD_COUNT]

    def _generate_dungeon(self):
        """Creates a maze using randomized DFS and places start/exit."""
        grid = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        stack = []
        start_pos = (1, 1)
        
        grid[start_pos[1], start_pos[0]] = 0
        stack.append(start_pos)
        
        visited_cells = {start_pos}
        farthest_cell = (start_pos, 0) # cell, distance

        while stack:
            x, y = stack[-1]
            
            # Find unvisited neighbors
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.GRID_WIDTH-1 and 0 < ny < self.GRID_HEIGHT-1 and (nx, ny) not in visited_cells:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                # Carve path
                grid[ny, nx] = 0
                grid[y + (ny - y) // 2, x + (nx - x) // 2] = 0
                
                visited_cells.add((nx, ny))
                stack.append((nx, ny))
                
                # Check for farthest cell for exit placement
                dist = len(stack)
                if dist > farthest_cell[1]:
                    farthest_cell = ((nx, ny), dist)
            else:
                stack.pop()

        # Add some loops by removing a few walls
        for _ in range(self.GRID_WIDTH * self.GRID_HEIGHT // 20):
            rx = self.np_random.integers(1, self.GRID_WIDTH - 1)
            ry = self.np_random.integers(1, self.GRID_HEIGHT - 1)
            if grid[ry, rx] == 1:
                grid[ry, rx] = 0

        return grid, start_pos, farthest_cell[0]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action
        reward = -0.01  # Cost of living
        terminated = False
        
        player_action_taken = False
        
        # --- Player Action Phase ---
        if space_held: # Attack action
            player_action_taken = True
            target_pos = (self.player_pos[0] + self.last_move_dir[0], self.player_pos[1] + self.last_move_dir[1])
            
            # Find monster at target position
            target_monster = None
            for monster in self.monsters:
                if tuple(monster["pos"]) == target_pos:
                    target_monster = monster
                    break
            
            if target_monster:
                # Player attacks monster
                damage = self.np_random.integers(10, 16)
                target_monster["hp"] -= damage
                self.damage_flashes[id(target_monster)] = 3 # Flash for 3 frames
                self._create_particles(target_pos, self.COLOR_PLAYER_OUTLINE, 10)
                # SFX: Player attack hit

                if target_monster["hp"] <= 0:
                    reward += 1.0 # Monster defeated reward
                    self.gold += self.GOLD_VALUE
                    reward += self.GOLD_VALUE * 0.1 # Gold collection reward
                    self.monsters.remove(target_monster)
                    self._create_particles(target_pos, self.COLOR_MONSTER, 30)
                    # SFX: Monster defeat
            else:
                # SFX: Player attack miss/whiff
                pass

        elif movement > 0: # Movement action
            player_action_taken = True
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.last_move_dir = (dx, dy)
            
            next_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            
            # Check for wall collision
            if self.grid[next_pos[1]][next_pos[0]] == 1:
                # SFX: Bump into wall
                pass
            # Check for monster collision
            elif any(tuple(m["pos"]) == next_pos for m in self.monsters):
                # SFX: Bump into monster
                pass
            else:
                self.player_pos = next_pos
                # SFX: Player footstep
        
        if player_action_taken:
            # Check for gold collection
            if self.player_pos in self.gold_piles:
                self.gold_piles.remove(self.player_pos)
                self.gold += self.GOLD_VALUE
                reward += self.GOLD_VALUE * 0.1
                self._create_particles(self.player_pos, self.COLOR_GOLD, 15)
                # SFX: Collect gold

            # Check for exit
            if self.player_pos == self.exit_pos:
                reward += 100.0
                self.level += 1
                self._setup_level() # Regenerate level
                # SFX: Level complete
                # Don't terminate, just go to next level
                
        # --- Monster Action Phase ---
        if player_action_taken:
            for monster in self.monsters:
                px, py = self.player_pos
                mx, my = monster["pos"]
                
                # Check if adjacent to player
                if abs(px - mx) + abs(py - my) == 1:
                    # Attack player
                    damage = self.np_random.integers(5, 10)
                    self.player_hp -= damage
                    self.damage_flashes[id(self)] = 3 # Flash player
                    self._create_particles(self.player_pos, self.COLOR_PLAYER, 10)
                    # SFX: Player takes damage
                else:
                    # Move randomly
                    possible_moves = []
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        next_m_pos = (mx + dx, my + dy)
                        if self.grid[next_m_pos[1]][next_m_pos[0]] == 0 and \
                           not any(tuple(m["pos"]) == next_m_pos for m in self.monsters) and \
                           next_m_pos != self.player_pos:
                            possible_moves.append(next_m_pos)
                    
                    if possible_moves:
                        monster["pos"] = list(self.np_random.choice(possible_moves, axis=0))

        # --- Termination Check ---
        self.steps += 1
        if self.player_hp <= 0:
            reward = -100.0
            terminated = True
            self.game_over = True
            # SFX: Player death
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _create_particles(self, pos, color, count):
        grid_center_x = pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2
        grid_center_y = pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            p = {
                "x": grid_center_x,
                "y": grid_center_y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": self.np_random.integers(15, 30),
                "color": color
            }
            self.particles.append(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if self.grid[y][x] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)

        # Draw exit
        ex, ey = self.exit_pos
        exit_rect = (ex * self.TILE_SIZE, ey * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        pygame.gfxdraw.filled_circle(self.screen, 
                                     int(ex * self.TILE_SIZE + self.TILE_SIZE/2), 
                                     int(ey * self.TILE_SIZE + self.TILE_SIZE/2),
                                     int(self.TILE_SIZE/4), (255, 255, 255, 50))

        # Draw gold
        for gx, gy in self.gold_piles:
            pygame.gfxdraw.filled_circle(self.screen, 
                                         int(gx * self.TILE_SIZE + self.TILE_SIZE / 2),
                                         int(gy * self.TILE_SIZE + self.TILE_SIZE / 2),
                                         int(self.TILE_SIZE / 4), self.COLOR_GOLD)

        # Update and draw particles
        for p in self.particles[:]:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p["life"] / 20))))
                temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
                pygame.draw.rect(temp_surf, (*p["color"], alpha), (0, 0, 3, 3))
                self.screen.blit(temp_surf, (int(p["x"]), int(p["y"])))

        # Draw monsters
        for m in self.monsters:
            mx, my = m["pos"]
            bob = math.sin(pygame.time.get_ticks() * 0.005 + mx) * 2
            m_rect = pygame.Rect(mx * self.TILE_SIZE + 4, my * self.TILE_SIZE + 4 + bob, self.TILE_SIZE - 8, self.TILE_SIZE - 8)
            
            flash_duration = self.damage_flashes[id(m)]
            color = (255, 255, 255) if flash_duration > 0 else self.COLOR_MONSTER
            outline_color = (255, 255, 255) if flash_duration > 0 else self.COLOR_MONSTER_OUTLINE
            if flash_duration > 0: self.damage_flashes[id(m)] -= 1

            pygame.draw.rect(self.screen, color, m_rect, border_radius=4)
            pygame.draw.rect(self.screen, outline_color, m_rect, width=2, border_radius=4)

        # Draw player
        px, py = self.player_pos
        bob = math.sin(pygame.time.get_ticks() * 0.005) * 2
        p_rect = pygame.Rect(px * self.TILE_SIZE + 4, py * self.TILE_SIZE + 4 + bob, self.TILE_SIZE - 8, self.TILE_SIZE - 8)
        
        flash_duration = self.damage_flashes[id(self)]
        color = (255, 255, 255) if flash_duration > 0 else self.COLOR_PLAYER
        outline_color = (255, 255, 255) if flash_duration > 0 else self.COLOR_PLAYER_OUTLINE
        if flash_duration > 0: self.damage_flashes[id(self)] -= 1

        pygame.draw.rect(self.screen, color, p_rect, border_radius=4)
        pygame.draw.rect(self.screen, outline_color, p_rect, width=2, border_radius=4)

    def _render_ui(self):
        # HP Bar
        hp_ratio = max(0, self.player_hp / self.PLAYER_MAX_HP)
        hp_bar_width = 200
        hp_bar_rect_bg = pygame.Rect(10, 10, hp_bar_width, 20)
        hp_bar_rect_fg = pygame.Rect(10, 10, int(hp_bar_width * hp_ratio), 20)
        pygame.draw.rect(self.screen, self.COLOR_HP_BAR_BG, hp_bar_rect_bg, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_HP_BAR, hp_bar_rect_fg, border_radius=5)
        hp_text = self.font_small.render(f"HP: {self.player_hp}/{self.PLAYER_MAX_HP}", True, self.COLOR_UI_TEXT)
        self.screen.blit(hp_text, (15, 12))

        # Gold Count
        gold_text = self.font_large.render(f"GOLD: {self.gold}", True, self.COLOR_GOLD)
        text_rect = gold_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(gold_text, text_rect)

        # Level indicator
        level_text = self.font_large.render(f"Dungeon Level: {self.level}", True, self.COLOR_UI_TEXT)
        text_rect = level_text.get_rect(midbottom=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 10))
        self.screen.blit(level_text, text_rect)

    def _get_info(self):
        # Using gold as the score metric
        return {"score": self.gold, "steps": self.steps, "level": self.level, "hp": self.player_hp}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        
        # Test game-specific assertions
        assert self.player_hp <= self.PLAYER_MAX_HP
        px, py = self.player_pos
        assert 0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Dungeon Crawler")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    
    # Map Pygame keys to MultiDiscrete actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    print(env.user_guide)

    while not terminated:
        movement_action = 0  # No-op
        space_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    movement_action = key_to_action[event.key]
                if event.key == pygame.K_SPACE:
                    space_action = 1
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    print("--- Game Reset ---")

        # Only step if an action was taken
        if movement_action != 0 or space_action != 0:
            action = [movement_action, space_action, 0] # Shift is not used
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Gold: {info['score']}, HP: {info['hp']}, Terminated: {terminated}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"--- GAME OVER ---")
            print(f"Final Score (Gold): {info['score']}, Final Level: {info['level']}")
            
    env.close()