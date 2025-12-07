import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:33:52.321267
# Source Brief: brief_00490.md
# Brief Index: 490
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Challenge an opponent in a game of cosmic kickball. Solve a match-3 puzzle to earn "
        "power-ups, then launch the ball and use special abilities to score."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor and aim. Press space to select/swap tiles and launch the ball. "
        "Use shift to arm the gravity-flip power-up."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_GRID = (50, 50, 80)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_LIGHT = (180, 220, 255)
    COLOR_OPPONENT = (255, 50, 50)
    COLOR_OPPONENT_LIGHT = (255, 180, 180)
    COLOR_BALL = (255, 255, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TRAJECTORY = (255, 255, 255, 150)
    
    TILE_COLORS = {
        1: (80, 200, 80),   # Green (Gravity)
        2: (220, 220, 50),  # Yellow (Shield)
        3: (180, 80, 255),  # Purple (Neutral)
        4: (100, 100, 100), # Grey (Neutral)
    }
    
    # Game parameters
    GRAVITY = pygame.math.Vector2(0, 0.3)
    SHIP_Y = SCREEN_HEIGHT // 2
    PLAYER_SHIP_X = 60
    OPPONENT_SHIP_X = SCREEN_WIDTH - 60
    BALL_RADIUS = 8
    SHIP_RADIUS = 25
    TILE_SIZE = 30
    GRID_ROWS = 3
    GRID_COLS = 3
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        self.font_huge = pygame.font.Font(None, 72)
        
        # Game state variables are initialized in reset()
        self.player_ship_pos = None
        self.opponent_ship_pos = None
        self.ball = None
        self.ball_velocity = None
        self.particles = None
        self.player_grid = None
        self.player_cursor = None
        self.player_selected_tile = None
        self.player_powerups = None
        self.opponent_powerups = None
        self.player_score = None
        self.opponent_score = None
        self.game_phase = None
        self.current_kicker = None
        self.aim_angle = None
        self.gravity_flip_armed = None
        self.round_end_timer = None
        self.steps = None
        self.last_space_press = False
        self.last_shift_press = False
        self.opponent_kick_speed_multiplier = 1.0
        self.ai_timer = 0
        self.ai_action_queue = deque()

        if render_mode == "human":
            pygame.display.set_caption("Cosmic Kickball")
            self.human_screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        self.render_mode = render_mode
        # self.reset() is called by the wrapper/runner, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.player_score = 0
        self.opponent_score = 0
        
        if options and 'opponent_kick_speed' in options:
             self.opponent_kick_speed_multiplier = options['opponent_kick_speed']

        self._reset_round()
        
        return self._get_observation(), self._get_info()

    def _reset_round(self):
        self.player_ship_pos = pygame.math.Vector2(self.PLAYER_SHIP_X, self.SHIP_Y)
        self.opponent_ship_pos = pygame.math.Vector2(self.OPPONENT_SHIP_X, self.SHIP_Y)
        
        self.ball = None
        self.ball_velocity = pygame.math.Vector2(0, 0)
        self.particles = []
        
        self.player_grid = self._create_tile_grid()
        self._check_and_resolve_matches(self.player_grid) # Ensure no matches at start
        self.player_cursor = [0, 0]
        self.player_selected_tile = None
        
        self.player_powerups = {"gravity": 0, "shield": 0}
        self.opponent_powerups = {"gravity": 0, "shield": 0}
        
        self.game_phase = "PLAYER_MATCH"
        self.current_kicker = "PLAYER"
        self.aim_angle = -math.pi / 4
        self.gravity_flip_armed = False
        self.round_end_timer = 0
        self.last_space_press = False
        self.last_shift_press = False
        self.ai_action_queue.clear()
        self.ai_timer = 0

    def step(self, action):
        movement, space_held, shift_held = action
        space_pressed = space_held and not self.last_space_press
        shift_pressed = shift_held and not self.last_shift_press
        self.last_space_press = bool(space_held)
        self.last_shift_press = bool(shift_held)
        
        reward = -0.001 # Small penalty for existing
        
        self._update_particles()

        if self.game_phase == "ROUND_OVER":
            self.round_end_timer -= 1
            if self.round_end_timer <= 0:
                if self.player_score >= 2 or self.opponent_score >= 2:
                    self.game_phase = "GAME_OVER"
                else:
                    self._reset_round()
        
        elif self.game_phase == "GAME_OVER":
            pass # Do nothing, wait for reset

        elif "PLAYER" in self.game_phase:
            reward += self._handle_player_turn(movement, space_pressed, shift_pressed)
        
        elif "OPPONENT" in self.game_phase:
            self._handle_opponent_turn()

        elif self.game_phase == "BALL_IN_FLIGHT":
            reward += self._update_ball_flight(space_pressed)

        self.steps += 1
        terminated = self._check_termination()
        truncated = False # No truncation condition in this game
        
        if terminated:
            if self.player_score > self.opponent_score:
                reward += 50
            elif self.opponent_score > self.player_score:
                reward -= 50
        
        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_to_human_screen(obs)
            self.clock.tick(self.FPS)

        return obs, reward, terminated, truncated, info

    def _handle_player_turn(self, movement, space_pressed, shift_pressed):
        reward = 0
        if self.game_phase == "PLAYER_MATCH":
            # Movement
            if movement == 1: self.player_cursor[1] = max(0, self.player_cursor[1] - 1) # Up
            elif movement == 2: self.player_cursor[1] = min(self.GRID_ROWS - 1, self.player_cursor[1] + 1) # Down
            elif movement == 3: self.player_cursor[0] = max(0, self.player_cursor[0] - 1) # Left
            elif movement == 4: self.player_cursor[0] = min(self.GRID_COLS - 1, self.player_cursor[0] + 1) # Right
            
            # Action
            if space_pressed:
                if self.player_selected_tile is None:
                    self.player_selected_tile = list(self.player_cursor)
                else:
                    # Swap tiles
                    x1, y1 = self.player_selected_tile
                    x2, y2 = self.player_cursor
                    if abs(x1 - x2) + abs(y1 - y2) == 1: # Adjacent check
                        self.player_grid[y1][x1], self.player_grid[y2][x2] = self.player_grid[y2][x2], self.player_grid[y1][x1]
                        
                        matches = self._check_and_resolve_matches(self.player_grid)
                        if matches:
                            # Successful match
                            num_matched = sum(len(m) for m in matches)
                            reward += 0.1 if num_matched == 3 else 0.5
                            for match in matches:
                                tile_type = match[0][2]
                                if tile_type == 1: self.player_powerups["gravity"] = min(3, self.player_powerups["gravity"] + 1)
                                if tile_type == 2: self.player_powerups["shield"] = min(3, self.player_powerups["shield"] + 1)
                                for x, y, _ in match:
                                    self._create_particles(self._get_tile_screen_pos(x, y), self.TILE_COLORS[tile_type], 10)
                            
                            self._refill_tiles(self.player_grid)
                            self.game_phase = "PLAYER_AIM"
                        else:
                            # Failed swap, swap back
                            self.player_grid[y1][x1], self.player_grid[y2][x2] = self.player_grid[y2][x2], self.player_grid[y1][x1]
                    
                    self.player_selected_tile = None
        
        elif self.game_phase == "PLAYER_AIM":
            # Aiming
            if movement == 1: self.aim_angle = max(-math.pi / 2 + 0.1, self.aim_angle - 0.05)
            elif movement == 2: self.aim_angle = min(-0.1, self.aim_angle + 0.05)
            
            # Arm powerup
            if shift_pressed and self.player_powerups["gravity"] > 0:
                self.gravity_flip_armed = not self.gravity_flip_armed

            # Launch
            if space_pressed:
                self.ball = self.player_ship_pos.copy()
                speed = 10
                self.ball_velocity = pygame.math.Vector2(speed * math.cos(self.aim_angle), speed * math.sin(self.aim_angle))
                self.game_phase = "BALL_IN_FLIGHT"
                self.current_kicker = "PLAYER"
                self._create_particles(self.player_ship_pos, self.COLOR_PLAYER_LIGHT, 20, 5)

        return reward

    def _handle_opponent_turn(self):
        self.ai_timer -= 1
        if self.ai_timer > 0:
            return

        if not self.ai_action_queue:
            self._plan_ai_turn()

        if self.ai_action_queue:
            action, duration = self.ai_action_queue.popleft()
            action()
            self.ai_timer = duration

    def _plan_ai_turn(self):
        # AI "thinks" for a moment
        self.ai_action_queue.append((lambda: None, 20)) 

        # For simplicity, AI gets a free powerup charge and launches immediately
        # A more complex AI would interact with its own tile grid.
        self.opponent_powerups["gravity"] = min(3, self.opponent_powerups["gravity"] + 1)
        
        def ai_launch():
            self.ball = self.opponent_ship_pos.copy()
            ai_angle = -math.pi * (3/4) # Aim towards player
            ai_angle += self.np_random.uniform(-0.2, 0.2) # Add some variance
            speed = 10 * self.opponent_kick_speed_multiplier
            self.ball_velocity = pygame.math.Vector2(speed * math.cos(ai_angle), speed * math.sin(ai_angle))
            self.game_phase = "BALL_IN_FLIGHT"
            self.current_kicker = "OPPONENT"
            self._create_particles(self.opponent_ship_pos, self.COLOR_OPPONENT_LIGHT, 20, 5)

        self.ai_action_queue.append((ai_launch, 1))

    def _update_ball_flight(self, space_pressed):
        reward = 0
        if self.ball is None: return reward

        # Gravity flip logic
        if self.current_kicker == "PLAYER" and self.gravity_flip_armed and space_pressed:
            self.GRAVITY.y *= -1
            self.player_powerups["gravity"] -= 1
            self.gravity_flip_armed = False
            self._create_screen_flash((200, 100, 255))
        
        self.ball_velocity += self.GRAVITY
        self.ball += self.ball_velocity

        # Collision checks
        hit_player = self.ball.distance_to(self.player_ship_pos) < self.BALL_RADIUS + self.SHIP_RADIUS
        hit_opponent = self.ball.distance_to(self.opponent_ship_pos) < self.BALL_RADIUS + self.SHIP_RADIUS

        if hit_player:
            self._create_particles(self.ball, self.COLOR_PLAYER, 50, 8)
            if self.player_powerups["shield"] > 0:
                self.player_powerups["shield"] -= 1
                # Ball bounces off
                self.ball_velocity.x *= -1.1 # slight speed boost on bounce
                self.ball += self.ball_velocity # move it out of collision
            else:
                self.opponent_score += 1
                reward -= 1.0
                self.game_phase = "ROUND_OVER"
                self.round_end_timer = self.FPS * 2 # 2 seconds
            self.ball = None
            if self.GRAVITY.y < 0: self.GRAVITY.y *= -1 # Reset gravity

        elif hit_opponent:
            self._create_particles(self.ball, self.COLOR_OPPONENT, 50, 8)
            if self.opponent_powerups["shield"] > 0:
                 self.opponent_powerups["shield"] -= 1
                 self.ball_velocity.x *= -1.1
                 self.ball += self.ball_velocity
            else:
                self.player_score += 1
                reward += 5.0
                self.game_phase = "ROUND_OVER"
                self.round_end_timer = self.FPS * 2 # 2 seconds
            self.ball = None
            if self.GRAVITY.y < 0: self.GRAVITY.y *= -1 # Reset gravity

        # Boundary checks
        elif self.ball and not (0 < self.ball.x < self.SCREEN_WIDTH and 0 < self.ball.y < self.SCREEN_HEIGHT):
            if self.current_kicker == "PLAYER":
                self.game_phase = "OPPONENT_TURN"
            else:
                self.game_phase = "PLAYER_MATCH"
            self.ball = None
            if self.GRAVITY.y < 0: self.GRAVITY.y *= -1 # Reset gravity
        
        return reward

    def _create_tile_grid(self):
        return [[self.np_random.integers(1, len(self.TILE_COLORS) + 1) for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]

    def _check_and_resolve_matches(self, grid):
        matches = []
        to_remove = set()

        # Check rows
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if grid[r][c] == grid[r][c+1] == grid[r][c+2] != 0:
                    match = {(c, r, grid[r][c]), (c+1, r, grid[r][c+1]), (c+2, r, grid[r][c+2])}
                    to_remove.update([(c, r), (c+1, r), (c+2, r)])
                    matches.append(list(match))

        # Check columns
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if grid[r][c] == grid[r+1][c] == grid[r+2][c] != 0:
                    match = {(c, r, grid[r][c]), (c, r+1, grid[r+1][c]), (c, r+2, grid[r+2][c])}
                    to_remove.update([(c, r), (c, r+1), (c, r+2)])
                    matches.append(list(match))
        
        for c, r in to_remove:
            grid[r][c] = 0
            
        return matches

    def _refill_tiles(self, grid):
        for c in range(self.GRID_COLS):
            empty_count = 0
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if grid[r][c] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    grid[r + empty_count][c] = grid[r][c]
                    grid[r][c] = 0
            
            for r in range(empty_count):
                grid[r][c] = self.np_random.integers(1, len(self.TILE_COLORS) + 1)
        
        # Recursively check for new matches after refill
        new_matches = self._check_and_resolve_matches(grid)
        if new_matches:
            self._refill_tiles(grid)

    def _check_termination(self):
        return self.player_score >= 2 or self.opponent_score >= 2 or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {
            "player_score": self.player_score,
            "opponent_score": self.opponent_score,
            "steps": self.steps,
            "phase": self.game_phase
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Simple starfield
        for _ in range(50):
            x = self.np_random.integers(0, self.SCREEN_WIDTH)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT)
            size = self.np_random.integers(1, 3)
            pygame.draw.rect(self.screen, (200, 200, 220), (x, y, size, size))

    def _render_game(self):
        # Render player ship
        self._render_ship(self.player_ship_pos, self.COLOR_PLAYER, self.COLOR_PLAYER_LIGHT, self.player_powerups["shield"] > 0)
        # Render opponent ship
        self._render_ship(self.opponent_ship_pos, self.COLOR_OPPONENT, self.COLOR_OPPONENT_LIGHT, self.opponent_powerups["shield"] > 0)

        # Render particles
        for p in self.particles:
            p[0] += p[1] # pos += vel
            p[2] -= 0.5 # lifetime
            if p[2] > 0:
                alpha = max(0, min(255, int(p[2] * 20)))
                self._draw_circle_alpha(self.screen, p[3] + (alpha,), p[0], int(p[2]))

        # Render ball
        if self.ball:
            # Glow effect
            for i in range(4, 0, -1):
                alpha = 100 - i * 20
                pygame.gfxdraw.filled_circle(self.screen, int(self.ball.x), int(self.ball.y), self.BALL_RADIUS + i, self.COLOR_BALL + (alpha,))
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball.x), int(self.ball.y), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, int(self.ball.x), int(self.ball.y), self.BALL_RADIUS, self.COLOR_BALL)

        # Render trajectory line for player aim phase
        if self.game_phase == "PLAYER_AIM":
            self._render_trajectory()

    def _render_ship(self, pos, color, light_color, has_shield):
        # Main body
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.SHIP_RADIUS, color)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), self.SHIP_RADIUS, light_color)
        # Cockpit
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.SHIP_RADIUS // 2, light_color)
        # Shield effect
        if has_shield:
            shield_alpha = 100 + math.sin(self.steps * 0.2) * 50
            self._draw_circle_alpha(self.screen, light_color + (int(shield_alpha),), pos, self.SHIP_RADIUS + 5, 3)

    def _render_trajectory(self):
        pos = self.player_ship_pos.copy()
        vel = pygame.math.Vector2(10 * math.cos(self.aim_angle), 10 * math.sin(self.aim_angle))
        temp_gravity = self.GRAVITY.copy()
        if self.gravity_flip_armed:
            temp_gravity.y *= -1

        for i in range(30):
            vel += temp_gravity
            pos += vel
            if i % 3 == 0:
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 2, self.COLOR_TRAJECTORY)

    def _render_ui(self):
        # Render scores
        player_score_text = self.font_huge.render(str(self.player_score), True, self.COLOR_PLAYER_LIGHT)
        self.screen.blit(player_score_text, (self.SCREEN_WIDTH // 4 - player_score_text.get_width() // 2, 20))
        
        opponent_score_text = self.font_huge.render(str(self.opponent_score), True, self.COLOR_OPPONENT_LIGHT)
        self.screen.blit(opponent_score_text, (self.SCREEN_WIDTH * 3 // 4 - opponent_score_text.get_width() // 2, 20))

        # Render player's tile grid
        self._render_tile_grid()
        
        # Render powerup status
        self._render_powerups()

        # Render Game Phase/End Message
        if self.game_phase == "ROUND_OVER" or self.game_phase == "GAME_OVER":
            msg = ""
            if self.player_score >= 2: msg = "YOU WIN!"
            elif self.opponent_score >= 2: msg = "YOU LOSE"
            
            if msg:
                text_surf = self.font_huge.render(msg, True, self.COLOR_TEXT)
                text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
                pygame.draw.rect(self.screen, self.COLOR_BG + (200,), text_rect.inflate(20, 20), border_radius=10)
                self.screen.blit(text_surf, text_rect)

    def _get_tile_screen_pos(self, c, r):
        grid_width = self.GRID_COLS * self.TILE_SIZE
        start_x = (self.SCREEN_WIDTH / 4) - (grid_width / 2)
        start_y = self.SCREEN_HEIGHT - (self.GRID_ROWS * self.TILE_SIZE) - 20
        return pygame.math.Vector2(start_x + c * self.TILE_SIZE, start_y + r * self.TILE_SIZE)

    def _render_tile_grid(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                pos = self._get_tile_screen_pos(c, r)
                rect = pygame.Rect(pos.x, pos.y, self.TILE_SIZE, self.TILE_SIZE)
                
                tile_id = self.player_grid[r][c]
                if tile_id > 0:
                    pygame.draw.rect(self.screen, self.TILE_COLORS[tile_id], rect.inflate(-4, -4), border_radius=4)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1, border_radius=4)

        # Draw cursor
        cursor_pos = self._get_tile_screen_pos(self.player_cursor[0], self.player_cursor[1])
        cursor_rect = pygame.Rect(cursor_pos.x, cursor_pos.y, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_LIGHT, cursor_rect, 2, border_radius=4)

        # Draw selection
        if self.player_selected_tile:
            sel_pos = self._get_tile_screen_pos(self.player_selected_tile[0], self.player_selected_tile[1])
            sel_rect = pygame.Rect(sel_pos.x, sel_pos.y, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, (255, 255, 255), sel_rect, 2, border_radius=4)

    def _render_powerups(self):
        # Gravity Flip Powerup
        grav_color = self.TILE_COLORS[1]
        grav_rect = pygame.Rect(self.SCREEN_WIDTH / 4 * 3 - 50, self.SCREEN_HEIGHT - 50, 40, 40)
        pygame.draw.rect(self.screen, grav_color, grav_rect, 0, border_radius=5)
        if self.gravity_flip_armed:
            pygame.draw.rect(self.screen, (255,255,255), grav_rect, 3, border_radius=5)
        
        charge_text = self.font_small.render(f"x{self.player_powerups['gravity']}", True, self.COLOR_TEXT)
        self.screen.blit(charge_text, (grav_rect.right + 5, grav_rect.centery - 10))
        
        # Shield Powerup
        shield_color = self.TILE_COLORS[2]
        shield_rect = pygame.Rect(self.SCREEN_WIDTH / 4 * 3 + 20, self.SCREEN_HEIGHT - 50, 40, 40)
        pygame.draw.rect(self.screen, shield_color, shield_rect, 0, border_radius=5)
        
        charge_text = self.font_small.render(f"x{self.player_powerups['shield']}", True, self.COLOR_TEXT)
        self.screen.blit(charge_text, (shield_rect.right + 5, shield_rect.centery - 10))

    def _create_particles(self, pos, color, count, max_speed=3):
        for _ in range(count):
            vel = pygame.math.Vector2(self.np_random.uniform(-max_speed, max_speed), self.np_random.uniform(-max_speed, max_speed))
            lifetime = self.np_random.uniform(5, 15)
            self.particles.append([pos.copy(), vel, lifetime, color])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 0]

    def _create_screen_flash(self, color):
        # This is a bit of a hack, we add a large, short-lived particle
        flash_particle = [pygame.math.Vector2(0,0), pygame.math.Vector2(0,0), 3, color]
        self.particles.append(flash_particle)
    
    def _draw_circle_alpha(self, surface, color, center, radius, width=0):
        target_rect = pygame.Rect(center, (0, 0)).inflate((radius * 2, radius * 2))
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(shape_surf, color, (radius, radius), radius, width)
        surface.blit(shape_surf, target_rect)

    def render(self):
        # This method is not used by the environment itself but is part of the standard Gym API
        return self._get_observation()

    def _render_to_human_screen(self, obs_array):
        # The obs_array is (H, W, C). Pygame needs (W, H) for surface, then transpose.
        surf = pygame.surfarray.make_surface(np.transpose(obs_array, (1, 0, 2)))
        self.human_screen.blit(surf, (0, 0))
        pygame.display.flip()

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Example of how to run the environment
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # --- Manual Play Controls ---
    # Arrow Keys: Move cursor / Aim
    # Space: Select / Swap / Launch / Use Gravity Flip
    # Left Shift: Arm Gravity Flip
    # R: Reset environment
    
    while not done:
        # Default action is "do nothing"
        action = [0, 0, 0] # move, space, shift
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT]: action[2] = 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Game Over! Final Score: Player {info['player_score']} - {info['opponent_score']} Opponent")
            print(f"Total Reward: {total_reward:.2f}")
            done = True

    env.close()