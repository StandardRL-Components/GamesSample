import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:34:59.300687
# Source Brief: brief_00498.md
# Brief Index: 498
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a futuristic, low-gravity curling game.
    Players match tiles to earn power-ups, then aim and release a stone,
    using the power-ups to influence its trajectory towards a target.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    game_description = (
        "Futuristic, low-gravity curling. Match tiles for power-ups, then aim and release your stone toward the target."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor or aim. Press space to select/swap tiles or launch the stone. Press shift to use power-ups."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    RINK_X_START = 200
    TILE_GRID_WIDTH = 180
    
    # Game States
    STATE_TILE_MATCHING = 0
    STATE_AIMING = 1
    STATE_ACTION = 2
    STATE_SCORING = 3
    STATE_AI_TURN = 4
    STATE_GAME_OVER = 5

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_RINK = (25, 35, 60)
    COLOR_GRID = (40, 50, 80)
    COLOR_TEXT = (220, 220, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)
    
    TILE_COLORS = {
        'blue': {'main': (50, 150, 255), 'glow': (100, 200, 255, 50)},
        'green': {'main': (50, 255, 150), 'glow': (100, 255, 200, 50)},
        'purple': {'main': (200, 100, 255), 'glow': (220, 150, 255, 50)},
    }
    TILE_TYPES = list(TILE_COLORS.keys())
    
    PLAYER_STONE_COLOR = (255, 200, 50)
    AI_STONE_COLOR = (255, 50, 100)
    
    HOUSE_COLORS = [(50, 80, 150), (60, 100, 180), (70, 120, 210)]

    # Game Parameters
    GRID_ROWS, GRID_COLS = 6, 5
    TILE_SIZE = 30
    TILE_GAP = 4
    MAX_STEPS = 2500
    SCORE_TO_WIN = 5
    AI_TURN_DURATION = 90 # steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        
        if self.render_mode == "human":
            pygame.display.set_caption("Space Curling")
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        else:
            self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # Persistent state (across matches)
        self.total_wins = 0
        self.ai_difficulty = 1.0 # 1.0 is pure random, 0.0 is perfect aim

        # State variables to be reset
        self.steps = 0
        self.score = 0
        self.opponent_score = 0
        self.game_over = False
        self.game_state = self.STATE_TILE_MATCHING
        self.current_turn = 'player' # 'player' or 'ai'
        
        self.player_stone = {}
        self.ai_stone = {}
        self.stones = []
        
        self.tile_grid = []
        self.cursor_pos = [0, 0]
        self.cursor_screen_pos = [0, 0]
        self.selected_tile = None
        self.swapping_tiles = []
        self.falling_tiles = []
        
        self.aim_angle = -math.pi / 4
        self.aim_power = 0.5 # 0 to 1
        
        self.powerups = {'gravity': 0, 'boost': 0}
        
        self.gravity_well = {
            'pos': (self.RINK_X_START + (self.SCREEN_WIDTH - self.RINK_X_START) * 0.8, self.SCREEN_HEIGHT / 2),
            'strength': 0.05,
            'is_repulsive': False,
            'active_steps': 0
        }
        
        self.particles = []
        self.stars = []
        
        self.was_space_held = False
        self.was_shift_held = False
        self.reward_this_step = 0

        self._generate_stars()
        # self.reset() is called by the wrapper, no need to call it here
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.opponent_score = 0
        self.game_over = False
        self.game_state = self.STATE_TILE_MATCHING
        self.current_turn = 'player'
        
        self.player_stone = {}
        self.ai_stone = {}
        self.stones = []
        
        self._generate_tile_grid()
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.cursor_screen_pos = list(self._get_tile_screen_pos(self.cursor_pos[0], self.cursor_pos[1]))
        self.selected_tile = None
        self.swapping_tiles = []
        self.falling_tiles = []

        self.aim_angle = -math.pi / 4
        self.aim_power = 0.5
        
        self.powerups = {'gravity': 0, 'boost': 0}
        self.gravity_well['active_steps'] = 0
        
        self.particles = []
        
        self.was_space_held = False
        self.was_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.was_space_held
        shift_pressed = shift_held and not self.was_shift_held
        
        if not self.game_over:
            if self.current_turn == 'player':
                self._handle_player_input(movement, space_pressed, shift_pressed)
            
            self._update_game_logic()

        self.was_space_held = space_held
        self.was_shift_held = shift_held

        terminated = self._check_termination()
        truncated = False # This environment does not truncate
        
        # Terminal rewards
        if terminated and not self.game_over:
            self.game_over = True
            self.game_state = self.STATE_GAME_OVER
            if self.score > self.opponent_score:
                self.reward_this_step += 10
                self.total_wins += 1
                if self.total_wins % 5 == 0:
                    self.ai_difficulty = max(0.0, self.ai_difficulty - 0.1)
            elif self.opponent_score > self.score:
                self.reward_this_step -= 10
        
        observation = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, self.reward_this_step, terminated, truncated, info

    def _handle_player_input(self, movement, space_pressed, shift_pressed):
        if self.game_state == self.STATE_TILE_MATCHING and not self.swapping_tiles and not self.falling_tiles:
            # Movement: Move cursor
            if movement == 1 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
            elif movement == 2 and self.cursor_pos[0] < self.GRID_ROWS - 1: self.cursor_pos[0] += 1
            elif movement == 3 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
            elif movement == 4 and self.cursor_pos[1] < self.GRID_COLS - 1: self.cursor_pos[1] += 1
            
            # Space: Select/Swap tile
            if space_pressed:
                # sound: tile_select.wav
                if not self.selected_tile:
                    self.selected_tile = list(self.cursor_pos)
                else:
                    cy, cx = self.cursor_pos
                    sy, sx = self.selected_tile
                    if abs(cy - sy) + abs(cx - sx) == 1:
                        # sound: tile_swap.wav
                        self.tile_grid[cy][cx], self.tile_grid[sy][sx] = self.tile_grid[sy][sx], self.tile_grid[cy][cx]
                        self.swapping_tiles = [(cy, cx), (sy, sx)]
                    self.selected_tile = None

        elif self.game_state == self.STATE_AIMING:
            # Movement: Adjust aim
            if movement == 3: self.aim_angle -= 0.05
            if movement == 4: self.aim_angle += 0.05
            if movement == 1: self.aim_power = min(1.0, self.aim_power + 0.02)
            if movement == 2: self.aim_power = max(0.1, self.aim_power - 0.02)
            self.aim_angle = self.aim_angle % (2 * math.pi)

            # Space: Release stone
            if space_pressed:
                # sound: stone_release.wav
                self.game_state = self.STATE_ACTION
                start_x = self.RINK_X_START + 30
                start_y = self.SCREEN_HEIGHT / 2
                max_speed = 10
                speed = self.aim_power * max_speed
                self.player_stone = self._create_stone(start_x, start_y, speed, self.aim_angle, self.PLAYER_STONE_COLOR, 'player')
                self.stones.append(self.player_stone)

        elif self.game_state == self.STATE_ACTION:
            # Shift: Use power-ups
            if shift_pressed:
                # For simplicity, use gravity first, then boost
                if self.powerups['gravity'] > 0:
                    # sound: gravity_activate.wav
                    self.powerups['gravity'] -= 1
                    self.gravity_well['active_steps'] = 150 # Active for 5 seconds
                    self.gravity_well['is_repulsive'] = self.np_random.choice([True, False])
                    self._spawn_particles(self.gravity_well['pos'], 30, (200, 100, 255), 2, 5)
                elif self.powerups['boost'] > 0 and self.player_stone:
                    # sound: boost_activate.wav
                    self.powerups['boost'] -= 1
                    vel_norm = math.sqrt(self.player_stone['vx']**2 + self.player_stone['vy']**2)
                    if vel_norm > 0.1:
                        boost_strength = 2.0
                        self.player_stone['vx'] += (self.player_stone['vx'] / vel_norm) * boost_strength
                        self.player_stone['vy'] += (self.player_stone['vy'] / vel_norm) * boost_strength
                        self.player_stone['boost_active'] = 60 # Visual effect duration

    def _update_game_logic(self):
        # --- Update animations and timed events ---
        self._update_particles()
        if self.gravity_well['active_steps'] > 0:
            self.gravity_well['active_steps'] -= 1
        for stone in self.stones:
            if stone.get('boost_active', 0) > 0:
                stone['boost_active'] -= 1

        # --- State Machine Logic ---
        if self.game_state == self.STATE_TILE_MATCHING:
            if self.swapping_tiles:
                # After a short delay, check for matches
                if not self._find_and_process_matches():
                    # No match, swap back
                    (y1, x1), (y2, x2) = self.swapping_tiles
                    self.tile_grid[y1][x1], self.tile_grid[y2][x2] = self.tile_grid[y2][x2], self.tile_grid[y1][x1]
                self.swapping_tiles.clear()
            
            if self.falling_tiles:
                # Animate tiles falling
                still_falling = False
                for tile_info in self.falling_tiles:
                    tile_info['y'] += 4
                    if tile_info['y'] >= tile_info['target_y']:
                        tile_info['y'] = tile_info['target_y']
                    else:
                        still_falling = True
                if not still_falling:
                    self._commit_falling_tiles()
                    self._find_and_process_matches() # Check for cascade matches
            
            # If board is settled and a match was made, transition state
            if not self.swapping_tiles and not self.falling_tiles and getattr(self, 'turn_made_match', False):
                self.game_state = self.STATE_AIMING
                self.turn_made_match = False
        
        elif self.game_state == self.STATE_ACTION:
            all_stopped = True
            for stone in self.stones:
                if stone['active']:
                    self._update_stone_physics(stone)
                    if math.sqrt(stone['vx']**2 + stone['vy']**2) > 0.05:
                        all_stopped = False
                    else:
                        stone['vx'], stone['vy'] = 0, 0
            
            if all_stopped and (self.player_stone or self.ai_stone):
                self.game_state = self.STATE_SCORING

        elif self.game_state == self.STATE_SCORING:
            self._calculate_round_score()
            self.stones.clear()
            self.player_stone, self.ai_stone = {}, {}
            
            if self._check_termination():
                self.game_state = self.STATE_GAME_OVER
                return

            if self.current_turn == 'player':
                self.current_turn = 'ai'
                self.game_state = self.STATE_AI_TURN
                self.ai_turn_timer = self.AI_TURN_DURATION
                self._run_ai_turn()
            else:
                self.current_turn = 'player'
                self.game_state = self.STATE_TILE_MATCHING
                self._generate_tile_grid()
        
        elif self.game_state == self.STATE_AI_TURN:
            self.ai_turn_timer -= 1
            if self.ai_stone and self.ai_stone['active']:
                self._update_stone_physics(self.ai_stone)
            
            ai_stopped = True
            if self.ai_stone and math.sqrt(self.ai_stone['vx']**2 + self.ai_stone['vy']**2) > 0.05:
                ai_stopped = False

            if self.ai_turn_timer <= 0 or ai_stopped:
                self.game_state = self.STATE_SCORING

    def _update_stone_physics(self, stone):
        # Gravity Well
        if self.gravity_well['active_steps'] > 0:
            dx = self.gravity_well['pos'][0] - stone['x']
            dy = self.gravity_well['pos'][1] - stone['y']
            dist_sq = dx**2 + dy**2
            if dist_sq > 1:
                force = self.gravity_well['strength'] / dist_sq
                force_dir = -1 if self.gravity_well['is_repulsive'] else 1
                stone['vx'] += force_dir * force * dx
                stone['vy'] += force_dir * force * dy

        # Friction
        stone['vx'] *= 0.995
        stone['vy'] *= 0.995
        
        # Update position
        stone['x'] += stone['vx']
        stone['y'] += stone['vy']
        
        # Boundary checks
        if stone['x'] < self.RINK_X_START + stone['r'] or stone['x'] > self.SCREEN_WIDTH - stone['r']:
            stone['vx'] *= -0.8
            stone['x'] = np.clip(stone['x'], self.RINK_X_START + stone['r'], self.SCREEN_WIDTH - stone['r'])
        if stone['y'] < stone['r'] or stone['y'] > self.SCREEN_HEIGHT - stone['r']:
            stone['vy'] *= -0.8
            stone['y'] = np.clip(stone['y'], stone['r'], self.SCREEN_HEIGHT - stone['r'])
        
        # Trail
        if self.steps % 2 == 0:
            self._spawn_particles( (stone['x'], stone['y']), 1, stone['color'], 1, 3, life=20, is_trail=True)

    def _find_and_process_matches(self):
        matches = self._find_matches()
        if not matches:
            return False

        # sound: match_found.wav
        self.turn_made_match = True
        num_matched_tiles = 0
        for r, c in matches:
            if self.tile_grid[r][c] is not None:
                num_matched_tiles +=1
                tile_type = self.tile_grid[r][c]['type']
                if tile_type == 'blue': self.powerups['gravity'] += 1
                elif tile_type == 'green': self.powerups['boost'] += 1
                
                # Particle effect
                px, py = self._get_tile_screen_pos(r, c)
                self._spawn_particles((px + self.TILE_SIZE/2, py + self.TILE_SIZE/2), 20, self.TILE_COLORS[tile_type]['main'], 2, 4)
                
                self.tile_grid[r][c] = None
        
        if num_matched_tiles >= 4:
            self.reward_this_step += 0.5
        elif num_matched_tiles > 0:
            self.reward_this_step += 0.1
        
        self._refill_tiles()
        return True

    def _calculate_round_score(self):
        house_center = self.gravity_well['pos']
        house_radii = [15, 40, 70]
        points = [3, 2, 1]

        player_dist = float('inf')
        if self.player_stone and self.player_stone.get('active'):
            player_dist = math.hypot(self.player_stone['x'] - house_center[0], self.player_stone['y'] - house_center[1])

        ai_dist = float('inf')
        if self.ai_stone and self.ai_stone.get('active'):
            ai_dist = math.hypot(self.ai_stone['x'] - house_center[0], self.ai_stone['y'] - house_center[1])

        if player_dist < ai_dist:
            for i, r in enumerate(house_radii):
                if player_dist <= r:
                    self.score += points[i]
                    self.reward_this_step += points[i]
                    # sound: score_player.wav
                    break
        elif ai_dist < player_dist:
            for i, r in enumerate(house_radii):
                if ai_dist <= r:
                    self.opponent_score += points[i]
                    self.reward_this_step -= points[i]
                    # sound: score_opponent.wav
                    break

    def _run_ai_turn(self):
        # AI makes its move instantly, animation plays out over AI_TURN_DURATION
        # Simple AI: random shot towards the house
        target_x = self.gravity_well['pos'][0] + self.np_random.uniform(-50, 50) * self.ai_difficulty
        target_y = self.gravity_well['pos'][1] + self.np_random.uniform(-50, 50) * self.ai_difficulty
        
        start_x = self.RINK_X_START + 30
        start_y = self.SCREEN_HEIGHT / 2
        
        angle = math.atan2(target_y - start_y, target_x - start_x)
        power = self.np_random.uniform(0.6, 0.9)
        speed = power * 10
        
        self.ai_stone = self._create_stone(start_x, start_y, speed, angle, self.AI_STONE_COLOR, 'ai')
        self.stones.append(self.ai_stone)

    def _check_termination(self):
        return self.steps >= self.MAX_STEPS or self.score >= self.SCORE_TO_WIN or self.opponent_score >= self.SCORE_TO_WIN

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_rink()
        self._render_tiles()
        self._render_stones()
        self._render_particles()
        if self.game_state == self.STATE_AIMING:
            self._render_aim_guide()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "opponent_score": self.opponent_score,
            "steps": self.steps,
            "wins": self.total_wins,
            "powerups": self.powerups,
        }

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        # The _get_observation method already draws everything to self.screen
        # So for human mode, we just need to update the display and tick the clock.
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.quit()

    # --- Helper methods for setup and game logic ---

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            x = self.np_random.integers(0, self.SCREEN_WIDTH)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT)
            size = self.np_random.uniform(0.5, 1.5)
            brightness = self.np_random.integers(50, 151)
            self.stars.append({'pos': [x, y], 'size': size, 'color': (brightness, brightness, brightness)})

    def _generate_tile_grid(self):
        self.tile_grid = [[self._create_tile(r, c) for c in range(self.GRID_COLS)] for r in range(self.GRID_ROWS)]
        while self._find_matches():
            self._find_and_process_matches()
        self.turn_made_match = False

    def _create_tile(self, r, c):
        tile_type = self.np_random.choice(self.TILE_TYPES)
        px, py = self._get_tile_screen_pos(r, c)
        return {'type': tile_type, 'x': px, 'y': py, 'target_y': py}
    
    def _create_stone(self, x, y, speed, angle, color, owner):
        return {
            'x': x, 'y': y, 'r': 12,
            'vx': speed * math.cos(angle),
            'vy': speed * math.sin(angle),
            'color': color, 'owner': owner, 'active': True
        }

    def _get_tile_screen_pos(self, r, c):
        x = self.TILE_GAP + c * (self.TILE_SIZE + self.TILE_GAP)
        y = (self.SCREEN_HEIGHT - self.GRID_ROWS * (self.TILE_SIZE + self.TILE_GAP)) / 2 + r * (self.TILE_SIZE + self.TILE_GAP)
        return x, y

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.tile_grid[r][c] is None: continue
                color = self.tile_grid[r][c]['type']
                # Horizontal
                if c < self.GRID_COLS - 2 and self.tile_grid[r][c+1] and self.tile_grid[r][c+2] and \
                   self.tile_grid[r][c+1]['type'] == color and self.tile_grid[r][c+2]['type'] == color:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < self.GRID_ROWS - 2 and self.tile_grid[r+1][c] and self.tile_grid[r+2][c] and \
                   self.tile_grid[r+1][c]['type'] == color and self.tile_grid[r+2][c]['type'] == color:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return list(matches)

    def _refill_tiles(self):
        self.falling_tiles = []
        for c in range(self.GRID_COLS):
            empty_count = 0
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.tile_grid[r][c] is None:
                    empty_count += 1
                elif empty_count > 0:
                    tile = self.tile_grid[r][c]
                    self.tile_grid[r + empty_count][c] = tile
                    self.tile_grid[r][c] = None
                    # Update target position for animation
                    _, target_y = self._get_tile_screen_pos(r + empty_count, c)
                    self.falling_tiles.append({'tile': tile, 'y': tile['y'], 'target_y': target_y, 'dest_r': r + empty_count, 'dest_c': c})

            for i in range(empty_count):
                r = empty_count - 1 - i
                px, py_final = self._get_tile_screen_pos(r, c)
                py_start = py_final - empty_count * (self.TILE_SIZE + self.TILE_GAP)
                new_tile = self._create_tile(r, c)
                new_tile['y'] = py_start
                self.falling_tiles.append({'tile': new_tile, 'y': py_start, 'target_y': py_final, 'dest_r': r, 'dest_c': c})

    def _commit_falling_tiles(self):
        for tile_info in self.falling_tiles:
            r, c = tile_info['dest_r'], tile_info['dest_c']
            self.tile_grid[r][c] = tile_info['tile']
            px, py = self._get_tile_screen_pos(r, c)
            self.tile_grid[r][c]['x'], self.tile_grid[r][c]['y'] = px, py
        self.falling_tiles.clear()

    def _spawn_particles(self, pos, count, color, min_speed, max_speed, life=30, is_trail=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            p = {
                'x': pos[0], 'y': pos[1],
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': life, 'max_life': life, 'color': color,
                'is_trail': is_trail
            }
            self.particles.append(p)

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            if not p['is_trail']:
                p['vx'] *= 0.95
                p['vy'] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
    
    # --- Rendering Methods ---

    def _render_background(self):
        for star in self.stars:
            star['pos'][0] = (star['pos'][0] - star['size'] * 0.05) % self.SCREEN_WIDTH
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['size'])

    def _render_rink(self):
        rink_rect = pygame.Rect(self.RINK_X_START, 0, self.SCREEN_WIDTH - self.RINK_X_START, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_RINK, rink_rect)
        
        # House (target)
        center = self.gravity_well['pos']
        radii = [70, 40, 15]
        for i, r in enumerate(radii):
            pygame.gfxdraw.filled_circle(self.screen, int(center[0]), int(center[1]), r, self.HOUSE_COLORS[i])
            pygame.gfxdraw.aacircle(self.screen, int(center[0]), int(center[1]), r, self.COLOR_GRID)

        # Gravity well visual
        if self.gravity_well['active_steps'] > 0:
            alpha = int(150 * (self.gravity_well['active_steps'] / 150))
            color = (220, 120, 255, alpha) if not self.gravity_well['is_repulsive'] else (255, 120, 120, alpha)
            s = pygame.Surface((140, 140), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, 70, 70, 70, color)
            self.screen.blit(s, (int(center[0] - 70), int(center[1] - 70)))

    def _render_tiles(self):
        # Draw falling tiles first
        for tile_info in self.falling_tiles:
            tile = tile_info['tile']
            color_data = self.TILE_COLORS[tile['type']]
            rect = pygame.Rect(tile['x'], tile_info['y'], self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, color_data['main'], rect, border_radius=5)
        
        # Draw grid tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.tile_grid[r][c] and not any(t['dest_r'] == r and t['dest_c'] == c for t in self.falling_tiles):
                    tile = self.tile_grid[r][c]
                    color_data = self.TILE_COLORS[tile['type']]
                    rect = pygame.Rect(tile['x'], tile['y'], self.TILE_SIZE, self.TILE_SIZE)
                    pygame.draw.rect(self.screen, color_data['main'], rect, border_radius=5)
        
        # Cursor and selection highlight
        if self.game_state == self.STATE_TILE_MATCHING:
            # Smooth cursor movement
            tx, ty = self._get_tile_screen_pos(self.cursor_pos[0], self.cursor_pos[1])
            self.cursor_screen_pos[0] += (tx - self.cursor_screen_pos[0]) * 0.4
            self.cursor_screen_pos[1] += (ty - self.cursor_screen_pos[1]) * 0.4

            cursor_rect = pygame.Rect(self.cursor_screen_pos[0]-2, self.cursor_screen_pos[1]-2, self.TILE_SIZE+4, self.TILE_SIZE+4)
            pygame.draw.rect(self.screen, self.PLAYER_STONE_COLOR, cursor_rect, 2, border_radius=7)

            if self.selected_tile:
                sx, sy = self._get_tile_screen_pos(self.selected_tile[0], self.selected_tile[1])
                sel_rect = pygame.Rect(sx-2, sy-2, self.TILE_SIZE+4, self.TILE_SIZE+4)
                pygame.draw.rect(self.screen, (255, 255, 255), sel_rect, 2, border_radius=7)

    def _render_stones(self):
        for stone in self.stones:
            if stone['active']:
                pos = (int(stone['x']), int(stone['y']))
                # Glow effect
                if stone.get('boost_active', 0) > 0:
                    s = pygame.Surface((stone['r']*4, stone['r']*4), pygame.SRCALPHA)
                    alpha = 100 * (stone['boost_active'] / 60)
                    pygame.gfxdraw.filled_circle(s, stone['r']*2, stone['r']*2, stone['r']*2, stone['color'] + (int(alpha),))
                    self.screen.blit(s, (pos[0]-stone['r']*2, pos[1]-stone['r']*2))
                
                # Main stone
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], stone['r'], stone['color'])
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], stone['r'], (0,0,0))
                # Highlight
                pygame.gfxdraw.filled_circle(self.screen, pos[0]+3, pos[1]-3, stone['r']//3, (255,255,255,100))

    def _render_particles(self):
        for p in self.particles:
            alpha = p['life'] / p['max_life']
            radius = alpha * 3
            color = p['color']
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color + (int(alpha * 255),), (radius, radius), radius)
            self.screen.blit(s, (int(p['x']-radius), int(p['y']-radius)))

    def _render_aim_guide(self):
        start_x = self.RINK_X_START + 30
        start_y = self.SCREEN_HEIGHT / 2
        length = 50 + self.aim_power * 150
        end_x = start_x + length * math.cos(self.aim_angle)
        end_y = start_y + length * math.sin(self.aim_angle)

        # Draw dotted line
        dx, dy = end_x - start_x, end_y - start_y
        dist = math.hypot(dx, dy)
        if dist == 0: return
        dx, dy = dx / dist, dy / dist
        for i in range(0, int(dist), 10):
            s = (start_x + i * dx, start_y + i * dy)
            e = (start_x + (i + 5) * dx, start_y + (i + 5) * dy)
            pygame.draw.line(self.screen, self.PLAYER_STONE_COLOR, s, e, 2)

    def _render_ui(self):
        # --- Draw text with a shadow for readability ---
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            main = font.render(text, True, color)
            self.screen.blit(shadow, (pos[0]+1, pos[1]+1))
            self.screen.blit(main, pos)

        # Scores
        player_score_text = f"PLAYER: {self.score}"
        ai_score_text = f"AI: {self.opponent_score}"
        draw_text(player_score_text, self.font_small, self.PLAYER_STONE_COLOR, (self.RINK_X_START + 10, 10))
        draw_text(ai_score_text, self.font_small, self.AI_STONE_COLOR, (self.SCREEN_WIDTH - self.font_small.size(ai_score_text)[0] - 10, 10))
        
        # Powerups
        draw_text("Power-ups:", self.font_small, self.COLOR_TEXT, (10, self.SCREEN_HEIGHT - 60))
        grav_text = f"Gravity Shift: {self.powerups['gravity']}"
        boost_text = f"Boost: {self.powerups['boost']}"
        draw_text(grav_text, self.font_small, self.TILE_COLORS['blue']['main'], (10, self.SCREEN_HEIGHT - 45))
        draw_text(boost_text, self.font_small, self.TILE_COLORS['green']['main'], (10, self.SCREEN_HEIGHT - 25))

        # Game Over message
        if self.game_state == self.STATE_GAME_OVER:
            result_text = "YOU WIN!" if self.score > self.opponent_score else "YOU LOSE"
            color = self.PLAYER_STONE_COLOR if self.score > self.opponent_score else self.AI_STONE_COLOR
            text_surf = self.font_large.render(result_text, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            pygame.draw.rect(self.screen, self.COLOR_BG, text_rect.inflate(20, 20))
            pygame.draw.rect(self.screen, self.COLOR_GRID, text_rect.inflate(20, 20), 2)
            self.screen.blit(text_surf, text_rect)

if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    terminated = False
    
    # --- Human player controls ---
    # Arrow keys: Move cursor / Aim
    # Space: Select tile / Release stone
    # Left Shift: Use power-up

    while not terminated:
        # Map keyboard inputs to action space
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
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}-{info['opponent_score']}")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING GAME ---")
                obs, info = env.reset()

    env.close()