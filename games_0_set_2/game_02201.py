
# Generated: 2025-08-28T04:03:10.391385
# Source Brief: brief_02201.md
# Brief Index: 2201

        
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
        "Controls: Arrow keys to move. Hold space to interact with objects (orbs, levers)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Explore a haunted house, collect spectral orbs, and find the exit before the ghost catches you or time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 25, 25
        self.MAX_STEPS = 1000
        self.TIME_LIMIT = 600
        self.NUM_ORBS = 3
        self.NUM_LEVERS = 1

        # Colors
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_WALL = (40, 30, 50)
        self.COLOR_WALL_TOP = (60, 50, 70)
        self.COLOR_FLOOR = (25, 20, 40)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_GHOST = (255, 50, 50)
        self.COLOR_ORB = (0, 150, 255)
        self.COLOR_EXIT = (50, 255, 50)
        self.COLOR_LEVER = (139, 69, 19)
        self.COLOR_DOOR = (100, 80, 120)
        self.COLOR_UI_TEXT = (220, 220, 220)

        # Isometric projection constants
        self.TILE_WIDTH_HALF = 18
        self.TILE_HEIGHT_HALF = 9
        self.TILE_Z = 18

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
        self.font = pygame.font.Font(None, 24)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = []
        self.player_pos = (0, 0)
        self.ghost_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.orbs = []
        self.levers = []
        self.doors = []
        self.orbs_collected = 0
        self.ghost_speed = 0.5
        self.ghost_move_accumulator = 0.0
        self.last_dist_to_orb = float('inf')
        self.last_dist_to_ghost = float('inf')

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self._validate_implementation()

    def _generate_map(self):
        """Generates a solvable maze using randomized DFS and places game elements."""
        self.grid = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        # Randomized DFS for maze generation
        stack = []
        start_pos = (self.np_random.integers(1, self.GRID_WIDTH // 2) * 2 - 1, self.np_random.integers(1, self.GRID_HEIGHT // 2) * 2 - 1)
        self.player_pos = start_pos
        self.grid[start_pos[1], start_pos[0]] = 0
        stack.append(start_pos)
        
        path = [start_pos]

        while stack:
            current_pos = stack[-1]
            x, y = current_pos
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                next_pos = self.np_random.choice(len(neighbors))
                nx, ny = neighbors[next_pos]
                
                self.grid[ny, nx] = 0
                self.grid[y + (ny - y) // 2, x + (nx - x) // 2] = 0
                
                path.append((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

        self.exit_pos = path[-1]
        
        empty_cells = list(zip(*np.where(self.grid == 0)))
        self.np_random.shuffle(empty_cells)
        
        # Place ghost away from player
        for cell in reversed(empty_cells):
            if math.dist(cell, self.player_pos) > 10:
                self.ghost_pos = cell
                empty_cells.remove(cell)
                break
        
        # Place orbs
        self.orbs = []
        for _ in range(self.NUM_ORBS):
            if empty_cells:
                pos = empty_cells.pop(0)
                if pos != self.player_pos and pos != self.exit_pos:
                    self.orbs.append(list(pos))

        # Place levers and doors
        self.levers = []
        self.doors = []
        for _ in range(self.NUM_LEVERS):
            if len(empty_cells) > 2:
                # Find a wall to place a door
                door_pos = None
                for _ in range(100): # Tries to find a good spot
                    cell = empty_cells[self.np_random.integers(0, len(empty_cells))]
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = cell[0] + dx, cell[1] + dy
                        if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny, nx] == 1:
                            door_pos = (nx, ny)
                            break
                    if door_pos:
                        break
                
                if door_pos:
                    lever_pos = empty_cells.pop(0)
                    self.levers.append({'pos': lever_pos, 'state': False}) # state: False=off, True=on
                    self.doors.append({'pos': door_pos, 'state': True}) # state: True=closed, False=open
                    self.grid[door_pos[1], door_pos[0]] = 2 # Mark door on grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.orbs_collected = 0
        self.ghost_speed = 0.5
        self.ghost_move_accumulator = 0.0
        
        self._generate_map()
        
        self.last_dist_to_orb = self._get_min_dist_to_orb()
        self.last_dist_to_ghost = math.dist(self.player_pos, self.ghost_pos)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        # shift_held is unused per brief
        
        reward = 0
        self.steps += 1
        
        # 1. Player Movement
        px, py = self.player_pos
        if movement == 1: # Up (Isometric North-East)
            py -= 1
        elif movement == 2: # Down (Isometric South-West)
            py += 1
        elif movement == 3: # Left (Isometric North-West)
            px -= 1
        elif movement == 4: # Right (Isometric South-East)
            px += 1

        # Check for valid move
        if 0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT:
            cell_type = self.grid[py, px]
            if cell_type == 0: # Empty space
                self.player_pos = (px, py)
            elif cell_type == 2: # Door
                for door in self.doors:
                    if door['pos'] == (px, py) and not door['state']: # if door is open
                        self.player_pos = (px, py)
                        break
        
        # 2. Player Interaction
        if space_held:
            # Collect orbs
            for orb_pos in self.orbs[:]:
                if self.player_pos == tuple(orb_pos):
                    self.orbs.remove(orb_pos)
                    self.orbs_collected += 1
                    reward += 10
                    # sfx: orb collection sound
                    break
            
            # Flip levers
            for i, lever in enumerate(self.levers):
                if self.player_pos == lever['pos']:
                    lever['state'] = not lever['state']
                    self.doors[i]['state'] = not self.doors[i]['state']
                    # sfx: lever switch sound
                    break

        # 3. Ghost Movement
        self.ghost_move_accumulator += self.ghost_speed
        if self.ghost_move_accumulator >= 1.0:
            self.ghost_move_accumulator -= 1.0
            self._move_ghost()
        
        # Increase ghost speed over time
        if self.steps > 0 and self.steps % 50 == 0:
            self.ghost_speed = min(1.0, self.ghost_speed + 0.1)

        # 4. Calculate continuous rewards
        new_dist_to_orb = self._get_min_dist_to_orb()
        if new_dist_to_orb < self.last_dist_to_orb:
            reward += 1
        self.last_dist_to_orb = new_dist_to_orb

        new_dist_to_ghost = math.dist(self.player_pos, self.ghost_pos)
        if new_dist_to_ghost < self.last_dist_to_ghost:
            reward -= 1 # Encourage moving away from ghost
        self.last_dist_to_ghost = new_dist_to_ghost

        # 5. Check termination conditions
        terminated = False
        # Win condition
        if self.orbs_collected == self.NUM_ORBS and self.player_pos == self.exit_pos:
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
        
        # Loss conditions
        if math.dist(self.player_pos, self.ghost_pos) < 1.0:
            reward -= 100
            self.score -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.TIME_LIMIT:
            reward -= 100
            self.score -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _move_ghost(self):
        """Simple greedy pathfinding for the ghost."""
        gx, gy = self.ghost_pos
        px, py = self.player_pos
        
        dx, dy = px - gx, py - gy
        
        # Determine primary and secondary move directions
        if abs(dx) > abs(dy):
            primary_move = (np.sign(dx), 0)
            secondary_move = (0, np.sign(dy))
        else:
            primary_move = (0, np.sign(dy))
            secondary_move = (np.sign(dx), 0)

        # Try primary move
        ngx, ngy = gx + primary_move[0], gy + primary_move[1]
        if self._is_walkable((ngx, ngy)):
            self.ghost_pos = (ngx, ngy)
            return

        # Try secondary move
        ngx, ngy = gx + secondary_move[0], gy + secondary_move[1]
        if self._is_walkable((ngx, ngy)):
            self.ghost_pos = (ngx, ngy)
            return

    def _is_walkable(self, pos):
        """Check if a grid position is walkable for the ghost."""
        x, y = pos
        if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
            return False
        return self.grid[int(y), int(x)] == 0 # Ghost can't open doors

    def _get_min_dist_to_orb(self):
        if not self.orbs:
            return math.dist(self.player_pos, self.exit_pos)
        return min(math.dist(self.player_pos, o) for o in self.orbs)

    def _grid_to_iso(self, x, y):
        """Converts grid coordinates to isometric screen coordinates."""
        iso_x = (x - y) * self.TILE_WIDTH_HALF
        iso_y = (x + y) * self.TILE_HEIGHT_HALF
        return iso_x, iso_y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Center camera on player
        cam_offset_x = self.WIDTH // 2 - self._grid_to_iso(*self.player_pos)[0]
        cam_offset_y = self.HEIGHT // 2 - self._grid_to_iso(*self.player_pos)[1] - self.TILE_Z

        # Sort all dynamic objects by their y-grid position for correct rendering order
        render_list = []
        for orb_pos in self.orbs:
            render_list.append(('orb', orb_pos))
        for lever in self.levers:
            render_list.append(('lever', lever['pos'], lever['state']))
        render_list.append(('player', self.player_pos))
        render_list.append(('ghost', self.ghost_pos))
        render_list.sort(key=lambda item: item[1][0] + item[1][1])

        # Render floor, walls, and exit
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                screen_x, screen_y = self._grid_to_iso(x, y)
                screen_x += cam_offset_x
                screen_y += cam_offset_y
                
                if self.grid[y, x] != 1: # If not a wall, draw floor
                    floor_points = [
                        (screen_x, screen_y + self.TILE_HEIGHT_HALF),
                        (screen_x + self.TILE_WIDTH_HALF, screen_y),
                        (screen_x, screen_y - self.TILE_HEIGHT_HALF),
                        (screen_x - self.TILE_WIDTH_HALF, screen_y)
                    ]
                    color = self.COLOR_EXIT if (x,y) == self.exit_pos and self.orbs_collected == self.NUM_ORBS else self.COLOR_FLOOR
                    pygame.gfxdraw.filled_polygon(self.screen, floor_points, color)

                # Draw walls and doors
                is_door = False
                door_state = False
                if self.grid[y,x] == 1 or self.grid[y,x] == 2:
                    if self.grid[y,x] == 2:
                        is_door = True
                        for door in self.doors:
                            if door['pos'] == (x,y):
                                door_state = door['state']
                                break
                    
                    if not is_door or door_state: # Draw if wall or closed door
                        wall_color = self.COLOR_DOOR if is_door else self.COLOR_WALL
                        wall_top_color = self.COLOR_DOOR if is_door else self.COLOR_WALL_TOP
                        
                        top_points = [
                            (screen_x, screen_y + self.TILE_HEIGHT_HALF),
                            (screen_x + self.TILE_WIDTH_HALF, screen_y),
                            (screen_x, screen_y - self.TILE_HEIGHT_HALF),
                            (screen_x - self.TILE_WIDTH_HALF, screen_y)
                        ]
                        bottom_points = [
                            (p[0], p[1] + self.TILE_Z) for p in top_points
                        ]
                        
                        # Draw sides
                        pygame.gfxdraw.filled_polygon(self.screen, [top_points[0], top_points[1], bottom_points[1], bottom_points[0]], wall_color)
                        pygame.gfxdraw.filled_polygon(self.screen, [top_points[1], top_points[2], bottom_points[2], bottom_points[1]], wall_color)
                        # Draw top
                        pygame.gfxdraw.filled_polygon(self.screen, top_points, wall_top_color)

        # Render dynamic objects from the sorted list
        for item in render_list:
            obj_type = item[0]
            pos = item[1]
            screen_x, screen_y = self._grid_to_iso(*pos)
            screen_x += cam_offset_x
            screen_y += cam_offset_y

            if obj_type == 'player':
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, (screen_x - 4, screen_y - 12, 8, 16))
            
            elif obj_type == 'ghost':
                # Fear aura
                aura_alpha = 30 + 20 * math.sin(self.steps * 0.2)
                aura_color = (*self.COLOR_GHOST, aura_alpha)
                radius = 30 + 5 * math.sin(self.steps * 0.2)
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, aura_color, (radius, radius), radius)
                self.screen.blit(temp_surf, (screen_x - radius, screen_y - radius))
                
                # Ghost body
                ghost_alpha = 150 + 50 * math.sin(self.steps * 0.3)
                pygame.draw.rect(self.screen, (*self.COLOR_GHOST, ghost_alpha), (screen_x - 5, screen_y - 15, 10, 20))

            elif obj_type == 'orb':
                # Pulsating effect
                pulse = 1 + 0.2 * math.sin(self.steps * 0.1 + pos[0])
                radius = int(5 * pulse)
                orb_color = self.COLOR_ORB
                pygame.draw.circle(self.screen, orb_color, (int(screen_x), int(screen_y)), radius)
                # Glow effect
                pygame.gfxdraw.filled_circle(self.screen, int(screen_x), int(screen_y), radius, (*orb_color, 100))
                
            elif obj_type == 'lever':
                state = item[2]
                lever_color = (200, 200, 0) if state else self.COLOR_LEVER
                pygame.draw.rect(self.screen, lever_color, (screen_x - 3, screen_y - 5, 6, 10))

    def _render_ui(self):
        # Orb count
        orb_text = self.font.render(f"Orbs: {self.orbs_collected}/{self.NUM_ORBS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(orb_text, (10, 10))
        
        # Timer
        time_left = max(0, self.TIME_LIMIT - self.steps)
        time_color = (255, 100, 100) if time_left < 100 else self.COLOR_UI_TEXT
        time_text = self.font.render(f"Time: {time_left}", True, time_color)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Game Over message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            message = ""
            if self.orbs_collected == self.NUM_ORBS and self.player_pos == self.exit_pos:
                message = "You Escaped!"
            elif math.dist(self.player_pos, self.ghost_pos) < 1.0:
                message = "Caught by the Ghost!"
            elif self.steps >= self.TIME_LIMIT:
                message = "Time's Up!"
            else:
                 message = "Game Over"

            end_text = pygame.font.Font(None, 50).render(message, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "orbs_collected": self.orbs_collected,
            "player_pos": self.player_pos,
            "ghost_pos": self.ghost_pos,
            "time_left": max(0, self.TIME_LIMIT - self.steps),
        }
        
    def close(self):
        pygame.quit()

    def _validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Haunted House Explorer")
    
    running = True
    total_reward = 0
    
    while running:
        # Pygame event handling
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Only step if an action is taken (since auto_advance is False)
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
            
            if terminated or truncated:
                print("Game Over! Resetting in 3 seconds...")
                pygame.time.wait(3000)
                obs, info = env.reset()
                total_reward = 0

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we don't need a fixed FPS clock loop for the env
        # But we need one for the display window to not freeze
        env.clock.tick(30)
        
    env.close()