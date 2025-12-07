import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:56:05.469995
# Source Brief: brief_00757.md
# Brief Index: 757
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a fast-paced, tile-matching action game.
    Players match colored tiles on a 10x10 grid to charge energy, then
    launch energy balls into the opponent's goal. The first to 5 goals wins.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) for tile swapping.
    - actions[1]: Space button (0=released, 1=held) to shoot.
    - actions[2]: Shift button (0=released, 1=held) to cycle aim direction.

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - A 640x400 RGB image of the game state.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "A fast-paced, tile-matching action game. Match colored tiles to charge energy, then "
        "launch energy balls into the opponent's goal."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor and swap tiles. "
        "Press space to shoot and shift to cycle your aim."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 10
    TILE_SIZE = 32
    BOARD_WIDTH = TILE_SIZE * GRID_SIZE
    BOARD_HEIGHT = TILE_SIZE * GRID_SIZE
    BOARD_X = (SCREEN_WIDTH - BOARD_WIDTH) // 2
    BOARD_Y = (SCREEN_HEIGHT - BOARD_HEIGHT) // 2

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 50)
    COLOR_GOAL = (50, 50, 70)
    COLOR_TEXT = (220, 220, 255)
    COLOR_TEXT_SHADOW = (10, 10, 20)
    TILE_COLORS = {
        1: (255, 50, 50),   # Red
        2: (50, 255, 50),   # Green
        3: (50, 100, 255),  # Blue
        4: (255, 255, 50),  # Yellow
    }
    PLAYER_CURSOR_COLOR = (255, 255, 255, 150)
    PLAYER_AIM_COLOR = (255, 255, 255)
    AI_AIM_COLOR = (255, 100, 100)

    # Game parameters
    MAX_ENERGY = 100
    ENERGY_PER_TILE = 5
    SHOOT_COST = 50
    MAX_SCORE = 5
    MAX_STEPS = 1000
    BALL_SPEED = 8
    INITIAL_AI_SPEED = 60 # Steps (2 seconds at 30 FPS)
    AI_SPEED_INCREASE_PER_GOAL = 1.5 # 0.05s * 30 FPS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # State variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.player_score = None
        self.opponent_score = None
        self.player_energy = None
        self.opponent_energy = None
        self.player_aim_dir = None
        self.balls = None
        self.particles = None
        self.steps = None
        self.game_over = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.ai_timer = None
        self.ai_match_speed = None
        self.combo_multiplier = None

        self.reset()
        # self.validate_implementation() # Removed for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.player_score = 0
        self.opponent_score = 0
        self.player_energy = 0
        self.opponent_energy = 0
        self.player_aim_dir = 0  # 0:up, 1:right, 2:down, 3:left
        self.balls = []
        self.particles = []
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.ai_match_speed = self.INITIAL_AI_SPEED
        self.ai_timer = self.ai_match_speed
        self.combo_multiplier = 0

        self._initialize_grid()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # 1. Handle player input
        reward += self._handle_player_input(movement, space_pressed, shift_pressed)

        # 2. Update AI
        reward += self._update_ai()

        # 3. Update game objects (balls, particles)
        reward += self._update_balls()
        self._update_particles()
        
        # 4. Check for termination conditions
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            if self.player_score >= self.MAX_SCORE:
                reward += 100 # Win bonus
            elif self.opponent_score >= self.MAX_SCORE:
                reward -= 100 # Loss penalty

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _initialize_grid(self):
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                self.grid[r, c] = self.np_random.integers(1, len(self.TILE_COLORS) + 1)
        
        # Ensure no initial matches
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            for r, c in matches:
                self.grid[r, c] = self.np_random.integers(1, len(self.TILE_COLORS) + 1)

    def _handle_player_input(self, movement, space_pressed, shift_pressed):
        reward = 0
        
        # Aiming
        if shift_pressed:
            self.player_aim_dir = (self.player_aim_dir + 1) % 4
            # sfx: aim_cycle.wav

        # Shooting
        if space_pressed and self.player_energy >= self.SHOOT_COST:
            self.player_energy -= self.SHOOT_COST
            self._fire_ball(is_player=True)
            # sfx: player_shoot.wav
        
        # Tile Swapping
        if movement != 0:
            cx, cy = self.cursor_pos
            nx, ny = cx, cy
            if movement == 1: ny -= 1 # Up
            elif movement == 2: ny += 1 # Down
            elif movement == 3: nx -= 1 # Left
            elif movement == 4: nx += 1 # Right
            
            self.cursor_pos = [nx, ny]
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)
            
            # Attempt swap with previous position
            prev_x, prev_y = cx, cy
            if (nx, ny) != (prev_x, prev_y):
                reward += self._attempt_swap(prev_y, prev_x, ny, nx)

        return reward

    def _attempt_swap(self, r1, c1, r2, c2):
        # Swap tiles
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        
        matches = self._find_all_matches()
        if not matches:
            # No match, swap back
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            return 0
        else:
            # Match found, resolve it
            # sfx: match_success.wav
            return self._resolve_matches(matches, is_player=True)

    def _update_ai(self):
        reward = 0
        self.ai_timer -= 1

        # AI Shooting
        if self.opponent_energy >= self.MAX_ENERGY:
            self.opponent_energy = 0
            self._fire_ball(is_player=False)
            # sfx: opponent_shoot.wav

        # AI Matching
        if self.ai_timer <= 0:
            self.ai_timer = self.ai_match_speed
            move = self._find_best_ai_move()
            if move:
                (r1, c1), (r2, c2) = move
                self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                matches = self._find_all_matches()
                if matches:
                    reward += self._resolve_matches(matches, is_player=False)
        return reward

    def _find_best_ai_move(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Try swapping right
                if c < self.GRID_SIZE - 1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if len(self._find_all_matches()) > 0:
                        self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                        return (r, c), (r, c+1)
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                # Try swapping down
                if r < self.GRID_SIZE - 1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if len(self._find_all_matches()) > 0:
                        self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                        return (r, c), (r+1, c)
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
        return None

    def _resolve_matches(self, matches, is_player):
        num_matched = len(matches)
        reward = 0
        self.combo_multiplier = max(0, num_matched - 3)
        
        # Award points and energy
        reward += num_matched * 0.1
        reward += self.combo_multiplier * 0.2
        energy_gain = num_matched * self.ENERGY_PER_TILE + self.combo_multiplier * self.ENERGY_PER_TILE

        if is_player:
            self.player_energy = min(self.MAX_ENERGY, self.player_energy + energy_gain)
        else:
            self.opponent_energy = min(self.MAX_ENERGY, self.opponent_energy + energy_gain)

        # Create particles and remove tiles
        for r, c in matches:
            tile_color = self.TILE_COLORS.get(self.grid[r,c], (255,255,255))
            px = self.BOARD_X + c * self.TILE_SIZE + self.TILE_SIZE // 2
            py = self.BOARD_Y + r * self.TILE_SIZE + self.TILE_SIZE // 2
            self._create_particles((px, py), tile_color, 20)
            self.grid[r, c] = 0 # Mark as empty

        # Handle gravity and refill
        self._apply_gravity_and_refill()
        
        # Check for chain reactions
        chain_matches = self._find_all_matches()
        if chain_matches:
            reward += self._resolve_matches(chain_matches, is_player)
        
        return reward

    def _find_all_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.add((r, c)); matches.add((r, c+1)); matches.add((r, c+2))
        # Vertical
        for r in range(self.GRID_SIZE - 2):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.add((r, c)); matches.add((r+1, c)); matches.add((r+2, c))
        return matches

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_row, c]
                    empty_row -= 1
        
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, len(self.TILE_COLORS) + 1)

    def _fire_ball(self, is_player):
        if is_player:
            x, y = self.BOARD_X - 20, self.SCREEN_HEIGHT // 2
            dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)] # Up, Right, Down, Left
            vel_x, vel_y = dirs[self.player_aim_dir]
            vel_x *= self.BALL_SPEED
            vel_y *= self.BALL_SPEED
            color = self.PLAYER_AIM_COLOR
        else: # Opponent
            x, y = self.BOARD_X + self.BOARD_WIDTH + 20, self.SCREEN_HEIGHT // 2
            vel_x, vel_y = -self.BALL_SPEED, 0
            color = self.AI_AIM_COLOR
        
        self.balls.append({'pos': [x, y], 'vel': [vel_x, vel_y], 'color': color, 'is_player': is_player})

    def _update_balls(self):
        reward = 0
        for ball in self.balls[:]:
            ball['pos'][0] += ball['vel'][0]
            ball['pos'][1] += ball['vel'][1]

            # Wall bounces
            if ball['pos'][1] < 0 or ball['pos'][1] > self.SCREEN_HEIGHT:
                ball['vel'][1] *= -1
                # sfx: bounce.wav

            # Goal scoring
            if ball['is_player'] and ball['pos'][0] > self.SCREEN_WIDTH:
                self.player_score += 1
                reward += 10
                self.balls.remove(ball)
                # sfx: player_goal.wav
            elif not ball['is_player'] and ball['pos'][0] < 0:
                self.opponent_score += 1
                self.ai_match_speed = max(15, self.ai_match_speed - self.AI_SPEED_INCREASE_PER_GOAL)
                reward -= 5
                self.balls.remove(ball)
                # sfx: opponent_goal.wav

            # Out of bounds (sides)
            elif ball['pos'][0] < -50 or ball['pos'][0] > self.SCREEN_WIDTH + 50:
                 self.balls.remove(ball)

        return reward

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # Drag
            p['vel'][1] *= 0.95
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.player_score >= self.MAX_SCORE or self.opponent_score >= self.MAX_SCORE:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw goals
        pygame.draw.rect(self.screen, self.COLOR_GOAL, (0, 0, 40, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_GOAL, (self.SCREEN_WIDTH - 40, 0, 40, self.SCREEN_HEIGHT))

        # Draw grid and tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = (self.BOARD_X + c * self.TILE_SIZE, self.BOARD_Y + r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
                tile_id = self.grid[r, c]
                if tile_id > 0:
                    color = self.TILE_COLORS[tile_id]
                    inner_rect = pygame.Rect(rect).inflate(-4, -4)
                    pygame.draw.rect(self.screen, color, inner_rect, border_radius=4)
        
        # Draw player cursor
        cx, cy = self.cursor_pos
        cursor_rect = (self.BOARD_X + cx * self.TILE_SIZE, self.BOARD_Y + cy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, self.PLAYER_CURSOR_COLOR, (0, 0, self.TILE_SIZE, self.TILE_SIZE), border_radius=6)
        self.screen.blit(s, cursor_rect)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(p['lifespan'] * 255 / 30)))
            color = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, color)

        # Draw balls
        for ball in self.balls:
            pos = (int(ball['pos'][0]), int(ball['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, ball['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, ball['color'])


    def _render_ui(self):
        # --- Scores ---
        self._draw_text(str(self.player_score), (60, 50), self.COLOR_TEXT, 32, center=True)
        self._draw_text(str(self.opponent_score), (self.SCREEN_WIDTH - 60, 50), self.COLOR_TEXT, 32, center=True)
        
        # --- Energy Bars ---
        # Player
        pygame.draw.rect(self.screen, self.COLOR_GRID, (20, 100, 40, self.SCREEN_HEIGHT - 200))
        energy_h = (self.SCREEN_HEIGHT - 200) * (self.player_energy / self.MAX_ENERGY)
        pygame.draw.rect(self.screen, self.PLAYER_AIM_COLOR, (20, 100 + (self.SCREEN_HEIGHT - 200) - energy_h, 40, energy_h))
        if self.player_energy >= self.SHOOT_COST:
            pygame.draw.rect(self.screen, (255,255,255), (20, 100, 40, self.SCREEN_HEIGHT - 200), 2)


        # Opponent
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.SCREEN_WIDTH - 60, 100, 40, self.SCREEN_HEIGHT - 200))
        energy_h = (self.SCREEN_HEIGHT - 200) * (self.opponent_energy / self.MAX_ENERGY)
        pygame.draw.rect(self.screen, self.AI_AIM_COLOR, (self.SCREEN_WIDTH - 60, 100 + (self.SCREEN_HEIGHT - 200) - energy_h, 40, energy_h))

        # --- Aim Indicator ---
        aim_start_x = self.BOARD_X - 20
        aim_start_y = self.SCREEN_HEIGHT // 2
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = dirs[self.player_aim_dir]
        aim_end_x = aim_start_x + dx * 20
        aim_end_y = aim_start_y + dy * 20
        pygame.draw.line(self.screen, self.PLAYER_AIM_COLOR, (aim_start_x, aim_start_y), (aim_end_x, aim_end_y), 2)
        
        # --- Combo text ---
        if self.combo_multiplier > 0:
            self._draw_text(f"COMBO x{self.combo_multiplier+3}", (self.SCREEN_WIDTH//2, self.BOARD_Y - 20), (255, 200, 0), 20, center=True)

    def _draw_text(self, text, pos, color, size, center=False):
        font = self.font_large if size >= 32 else self.font_small
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {"score": self.player_score, "opponent_score": self.opponent_score, "steps": self.steps}

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you need to unset the dummy video driver
    # and install pygame:
    # pip install pygame
    # unset SDL_VIDEODRIVER
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tile Match Combat")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    running = True
    
    # --- Manual Control Mapping ---
    # Arrows: Move cursor & swap
    # Space: Shoot
    # Left Shift: Change Aim
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}-{info['opponent_score']}")
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}-{info['opponent_score']}")
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata['render_fps'])

    env.close()