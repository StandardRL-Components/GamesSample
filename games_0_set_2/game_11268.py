import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:43:20.999432
# Source Brief: brief_01268.md
# Brief Index: 1268
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Rhythm Thief: A stealth-action puzzle game in a musical academy.

    The player must navigate through the academy, avoiding guards.
    By initiating a note-matching mini-game, the player can create musical
    combos that stun nearby guards. The goal is to reach the concert hall
    at the end of the level.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held) -> Initiates note-matching
    - actions[2]: Shift button (0=released, 1=held) -> Uses an item

    Observation Space: Box(0, 255, (400, 640, 3), uint8) -> RGB image of the game.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a musical academy, avoiding guards. Initiate rhythm mini-games to stun guards and reach the concert hall."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Press space to start a rhythm mini-game. Press shift to use an item or exit the mini-game."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000
    TILE_SIZE = 20

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_WALL = (40, 30, 60)
    COLOR_FLOOR = (25, 20, 40)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 150)
    COLOR_GUARD = (200, 50, 50)
    COLOR_GUARD_GLOW = (120, 30, 30)
    COLOR_VISION_CONE = (180, 80, 80, 50)
    COLOR_GOAL = (255, 223, 0)
    COLOR_ITEM = (50, 255, 50)
    COLOR_STUNNED = (200, 200, 255)
    COLOR_TEXT = (220, 220, 240)
    NOTE_COLORS = [(255, 255, 0), (255, 0, 255), (0, 255, 0), (255, 128, 0)]

    # Game Parameters
    PLAYER_SPEED = 0.25 # tiles per step
    GUARD_SPEED = 0.05 # tiles per step
    GUARD_VISION_RANGE = 6 # tiles
    GUARD_VISION_ANGLE = 45 # degrees
    GUARD_STUN_DURATION = 150 # steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_huge = pygame.font.SysFont("Consolas", 48, bold=True)

        # These attributes are defined in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_target_pos = np.array([0.0, 0.0])
        self.guards = []
        self.items_on_map = []
        self.inventory = []
        self.particles = []
        self.map_grid = np.array([])
        self.goal_pos = np.array([0, 0])
        self.is_mini_game_active = False
        self.mini_game = {}
        self.prev_space_held = False
        self.prev_shift_held = False

        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.is_mini_game_active = False
        self.prev_space_held = False
        self.prev_shift_held = False

        self._create_map()
        start_pos = self._get_random_floor_tile()
        self.player_pos = np.array(start_pos, dtype=float)
        self.player_target_pos = np.array(start_pos, dtype=float)

        self._spawn_guards(num_guards=3)
        self._spawn_items(num_items=2)
        
        self.inventory = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Action Handling ---
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Game Logic Update ---
        if self.is_mini_game_active:
            reward += self._update_mini_game(movement, space_pressed, shift_pressed)
        else:
            reward += self._update_main_game(movement, space_pressed, shift_pressed)
            
        self._update_player_position()
        self._update_guards()
        self._update_particles()
        
        # --- Check Game State ---
        termination_reward, terminated = self._check_termination()
        reward += termination_reward
        self.game_over = terminated
        self.steps += 1
        truncated = False
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            truncated = True
        
        self.score += reward

        return self._get_observation(), reward, self.game_over, truncated, self._get_info()

    def _update_main_game(self, movement, space_pressed, shift_pressed):
        reward = 0.0
        
        # Player Movement
        if movement != 0:
            new_target = self.player_target_pos.copy()
            if movement == 1: new_target[1] -= 1  # Up
            elif movement == 2: new_target[1] += 1  # Down
            elif movement == 3: new_target[0] -= 1  # Left
            elif movement == 4: new_target[0] += 1  # Right
            
            if self._is_walkable(int(new_target[0]), int(new_target[1])):
                self.player_target_pos = new_target

        # Item Collection
        player_grid_pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        for item in self.items_on_map[:]:
            if player_grid_pos == tuple(item['pos']):
                self.inventory.append(item['type'])
                self.items_on_map.remove(item)
                reward += 10.0
                # SFX: Item pickup sound
                self._create_particles(self.player_pos * self.TILE_SIZE, 20, self.COLOR_ITEM)

        # Use Item
        if shift_pressed and self.inventory:
            item_type = self.inventory.pop(0)
            if item_type == 'stun_orb':
                # Stun all guards on screen
                for guard in self.guards:
                    guard['state'] = 'stunned'
                    guard['stun_timer'] = self.GUARD_STUN_DURATION
                    # SFX: Mass stun sound
                self._create_particles(self.player_pos * self.TILE_SIZE, 30, self.COLOR_STUNNED, life=40)

        # Initiate Mini-game
        if space_pressed:
            self.is_mini_game_active = True
            self._init_mini_game()
            # SFX: Mini-game start chime
        
        # Base reward for survival
        reward += 0.1
        return reward

    def _update_mini_game(self, movement, space_pressed, shift_pressed):
        mg = self.mini_game
        
        # Exit mini-game
        if shift_pressed:
            self.is_mini_game_active = False
            return 0.0
            
        # Move cursor
        if movement != 0:
            if movement == 1 and mg['cursor'][1] > 0: mg['cursor'][1] -= 1
            elif movement == 2 and mg['cursor'][1] < 2: mg['cursor'][1] += 1
            elif movement == 3 and mg['cursor'][0] > 0: mg['cursor'][0] -= 1
            elif movement == 4 and mg['cursor'][0] < 2: mg['cursor'][0] += 1
        
        # Select note
        if space_pressed:
            cx, cy = mg['cursor']
            selected_note = mg['grid'][cy][cx]
            
            if selected_note not in mg['selection']:
                mg['selection'].append(selected_note)
                mg['selection_pos'].append((cx, cy))
            
            # Check for combo
            if len(mg['selection']) == 3:
                self.is_mini_game_active = False
                if mg['selection'][0]['type'] == mg['selection'][1]['type'] == mg['selection'][2]['type']:
                    # Successful combo!
                    self._stun_nearest_guard()
                    # SFX: Success chord
                    return 1.0
                else:
                    # Failed combo
                    # SFX: Dissonant chord
                    return -0.5
        return 0.0

    def _update_player_position(self):
        # Interpolate player position for smooth movement
        self.player_pos += (self.player_target_pos - self.player_pos) * self.PLAYER_SPEED

    def _update_guards(self):
        for guard in self.guards:
            if guard['state'] == 'stunned':
                guard['stun_timer'] -= 1
                if guard['stun_timer'] <= 0:
                    guard['state'] = 'patrolling'
                continue

            # Move guard along path
            target_waypoint = np.array(guard['path'][guard['path_index']], dtype=float)
            dist_to_waypoint = np.linalg.norm(target_waypoint - guard['pos'])

            if dist_to_waypoint < 0.1:
                guard['path_index'] = (guard['path_index'] + 1) % len(guard['path'])
            else:
                direction = (target_waypoint - guard['pos']) / dist_to_waypoint
                guard['pos'] += direction * self.GUARD_SPEED
                guard['angle'] = math.atan2(direction[1], direction[0])

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        # Win condition
        if self._is_on_tile(self.player_pos, self.goal_pos):
            return 100.0, True

        # Lose condition
        player_center = self.player_pos * self.TILE_SIZE + self.TILE_SIZE / 2
        for guard in self.guards:
            if guard['state'] != 'stunned':
                guard_center = guard['pos'] * self.TILE_SIZE + self.TILE_SIZE / 2
                dist = np.linalg.norm(player_center - guard_center)
                
                if dist < self.GUARD_VISION_RANGE * self.TILE_SIZE:
                    angle_to_player = math.atan2(player_center[1] - guard_center[1], player_center[0] - guard_center[0])
                    angle_diff = abs(math.degrees(self._normalize_angle(angle_to_player - guard['angle'])))
                    
                    if angle_diff < self.GUARD_VISION_ANGLE / 2:
                        if self._has_line_of_sight(guard['pos'], self.player_pos):
                            # SFX: Alert sound, game over music
                            return -100.0, True
        return 0.0, False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.is_mini_game_active:
            self._render_mini_game()
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "inventory": len(self.inventory)}

    # --- RENDER METHODS ---

    def _render_game(self):
        # Draw map, goal, and items
        for y, row in enumerate(self.map_grid):
            for x, tile in enumerate(row):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                if tile == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)
        
        goal_rect = (self.goal_pos[0] * self.TILE_SIZE, self.goal_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, goal_rect)
        pygame.gfxdraw.rectangle(self.screen, goal_rect, (*self.COLOR_GOAL, 150))
        
        for item in self.items_on_map:
            pos = (int((item['pos'][0] + 0.5) * self.TILE_SIZE), int((item['pos'][1] + 0.5) * self.TILE_SIZE))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_ITEM)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, self.COLOR_ITEM)

        # Draw particles
        self._render_particles()

        # Draw guards
        for guard in self.guards:
            self._render_guard(guard)

        # Draw player
        self._render_player()

    def _render_player(self):
        pos_px = (
            int(self.player_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2),
            int(self.player_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2)
        )
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], 12, (*self.COLOR_PLAYER_GLOW, 100))
        pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], 8, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], 8, self.COLOR_PLAYER)

    def _render_guard(self, guard):
        pos_px = (
            int(guard['pos'][0] * self.TILE_SIZE + self.TILE_SIZE / 2),
            int(guard['pos'][1] * self.TILE_SIZE + self.TILE_SIZE / 2)
        )
        
        # Vision Cone
        if guard['state'] != 'stunned':
            vision_range_px = self.GUARD_VISION_RANGE * self.TILE_SIZE
            angle_rad = guard['angle']
            half_angle_rad = math.radians(self.GUARD_VISION_ANGLE / 2)
            p1 = pos_px
            p2 = (pos_px[0] + vision_range_px * math.cos(angle_rad - half_angle_rad),
                  pos_px[1] + vision_range_px * math.sin(angle_rad - half_angle_rad))
            p3 = (pos_px[0] + vision_range_px * math.cos(angle_rad + half_angle_rad),
                  pos_px[1] + vision_range_px * math.sin(angle_rad + half_angle_rad))
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_VISION_CONE)
        
        # Body
        color = self.COLOR_STUNNED if guard['state'] == 'stunned' else self.COLOR_GUARD
        glow_color = (*self.COLOR_STUNNED, 100) if guard['state'] == 'stunned' else (*self.COLOR_GUARD_GLOW, 100)
        
        pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], 10, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], 7, color)
        pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], 7, color)

        if guard['state'] == 'stunned':
            text = self.font_small.render("Zzz", True, self.COLOR_STUNNED)
            self.screen.blit(text, (pos_px[0] - 10, pos_px[1] - 25))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), color)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        # Inventory
        inv_text = self.font_small.render("INVENTORY", True, self.COLOR_TEXT)
        self.screen.blit(inv_text, (10, self.HEIGHT - 40))
        for i, item_type in enumerate(self.inventory):
            if item_type == 'stun_orb':
                color = self.COLOR_STUNNED
            else:
                color = self.COLOR_ITEM
            pygame.draw.rect(self.screen, color, (10 + i * 25, self.HEIGHT - 25, 20, 20))

        # Game Over Text
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            player_on_goal = self._is_on_tile(self.player_pos, self.goal_pos)
            end_text_str = "VICTORY" if player_on_goal else "CAUGHT!"
            end_text = self.font_huge.render(end_text_str, True, self.COLOR_GOAL if player_on_goal else self.COLOR_GUARD)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _render_mini_game(self):
        # Overlay
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 220))
        self.screen.blit(s, (0,0))

        mg = self.mini_game
        grid_size = 180
        cell_size = grid_size / 3
        grid_x = (self.WIDTH - grid_size) / 2
        grid_y = (self.HEIGHT - grid_size) / 2

        # Draw grid and notes
        for r in range(3):
            for c in range(3):
                note = mg['grid'][r][c]
                note_pos = (int(grid_x + (c + 0.5) * cell_size), int(grid_y + (r + 0.5) * cell_size))
                
                is_selected = note in mg['selection']
                radius = int(cell_size * 0.4) if not is_selected else int(cell_size * 0.45)
                color = self.NOTE_COLORS[note['type']]
                
                pygame.gfxdraw.filled_circle(self.screen, note_pos[0], note_pos[1], radius, color)
                pygame.gfxdraw.aacircle(self.screen, note_pos[0], note_pos[1], radius, color)
        
        # Draw cursor
        cursor_pos = (int(grid_x + mg['cursor'][0] * cell_size), int(grid_y + mg['cursor'][1] * cell_size))
        cursor_rect = pygame.Rect(cursor_pos[0], cursor_pos[1], cell_size, cell_size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, cursor_rect, 2)

    # --- HELPER METHODS ---

    def _create_map(self):
        width_tiles, height_tiles = self.WIDTH // self.TILE_SIZE, self.HEIGHT // self.TILE_SIZE
        self.map_grid = np.ones((height_tiles, width_tiles), dtype=int)
        
        # Basic room carving
        for _ in range(15):
            w, h = self.np_random.integers(3, 8, size=2)
            x, y = self.np_random.integers(1, width_tiles - w - 1), self.np_random.integers(1, height_tiles - h - 1)
            self.map_grid[y:y+h, x:x+w] = 0
        
        # Set goal
        self.goal_pos = self._get_random_floor_tile()

    def _get_random_floor_tile(self):
        floor_tiles = np.argwhere(self.map_grid == 0)
        return self.np_random.choice(floor_tiles)[::-1] # return as (x,y)

    def _is_walkable(self, x, y):
        w, h = self.map_grid.shape[1], self.map_grid.shape[0]
        if 0 <= x < w and 0 <= y < h:
            return self.map_grid[y, x] == 0
        return False

    def _is_on_tile(self, pos, tile_pos):
        return int(pos[0]) == tile_pos[0] and int(pos[1]) == tile_pos[1]

    def _spawn_guards(self, num_guards):
        self.guards = []
        for _ in range(num_guards):
            path = []
            for _ in range(self.np_random.integers(2, 5)):
                path.append(self._get_random_floor_tile())
            
            if not path: continue

            self.guards.append({
                'pos': np.array(path[0], dtype=float),
                'path': path,
                'path_index': 0,
                'state': 'patrolling', # patrolling, stunned
                'stun_timer': 0,
                'angle': self.np_random.uniform(0, 2 * math.pi)
            })

    def _spawn_items(self, num_items):
        self.items_on_map = []
        for _ in range(num_items):
            self.items_on_map.append({
                'pos': self._get_random_floor_tile(),
                'type': 'stun_orb'
            })

    def _init_mini_game(self):
        self.mini_game = {
            'grid': [[{'type': self.np_random.integers(0, 4)} for _ in range(3)] for _ in range(3)],
            'cursor': [1, 1],
            'selection': [],
            'selection_pos': []
        }

    def _stun_nearest_guard(self):
        player_center = self.player_pos * self.TILE_SIZE
        min_dist = float('inf')
        nearest_guard = None
        for guard in self.guards:
            if guard['state'] != 'stunned':
                dist = np.linalg.norm(player_center - (guard['pos'] * self.TILE_SIZE))
                if dist < min_dist:
                    min_dist = dist
                    nearest_guard = guard
        
        if nearest_guard:
            nearest_guard['state'] = 'stunned'
            nearest_guard['stun_timer'] = self.GUARD_STUN_DURATION
            self._create_particles(nearest_guard['pos'] * self.TILE_SIZE, 30, self.COLOR_STUNNED)

    def _create_particles(self, pos, count, color, life=20, size=2):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': self.np_random.integers(life // 2, life),
                'max_life': life,
                'color': color,
                'size': size
            })

    def _normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _has_line_of_sight(self, start_pos, end_pos):
        # Bresenham's line algorithm to check for wall obstructions
        x0, y0 = int(start_pos[0]), int(start_pos[1])
        x1, y1 = int(end_pos[0]), int(end_pos[1])
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            if self.map_grid[y0, x0] == 1: return False # Hit a wall
            if x0 == x1 and y0 == y1: break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return True

    def render(self):
        # This method is not used by the gym interface but can be useful for human playing
        # The core rendering logic is in _get_observation()
        return self._get_observation()

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # Example of how to use the environment
    # This main block is for human play and is not used by the evaluation system.
    # It will not be run during testing, so you can leave it as-is.
    
    # Un-comment the line below to run with display
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11") 
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Human player controls ---
    # Arrow keys: Move
    # Space:      Enter/interact in mini-game
    # Shift:      Use item / Exit mini-game
    # R:          Reset
    # Q:          Quit
    
    # Check if we are in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Running in headless mode. No display will be shown.")
        # Simple loop to test the environment
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished! Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
                obs, info = env.reset()
        env.close()
    else:
        pygame.display.set_caption("Rhythm Thief")
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            movement = 0 # none
            space = 0
            shift = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                    if event.key == pygame.K_q:
                        running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"Episode finished! Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")

            # Display the observation
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(GameEnv.FPS)
            
        env.close()