import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:29:04.519049
# Source Brief: brief_00496.md
# Brief Index: 496
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Gymnasium environment for a gravity-bending netball game in a nebula.
    Players move, match tiles on the floor to gain power-ups, and shoot a
    ball into the opponent's goal, all while managing a switchable gravity field.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "A zero-gravity netball game set in a nebula. Match floor tiles for power-ups, flip gravity to outmaneuver your opponent, and score goals."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move. Press space to shoot or catch the ball. Use shift to flip the direction of gravity."
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- CRITICAL: Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.width, self.height = 640, 400
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game Constants ---
        self.PLAYER_COLOR = (0, 150, 255)
        self.OPPONENT_COLOR = (255, 50, 50)
        self.BALL_COLOR = (255, 150, 0)
        self.COLOR_BG = (10, 5, 25)
        self.TILE_COLORS = [(60, 60, 180), (180, 60, 60), (60, 180, 60)] # Blue, Red, Green
        self.POWERUP_COLORS = {'speed': (0, 255, 0), 'gravity': (180, 0, 255)}

        self.PLAYER_SIZE = 15
        self.BALL_SIZE = 8
        self.PLAYER_SPEED = 4.0
        self.BALL_MAX_SPEED = 12.0
        self.GRAVITY_STRENGTH = 0.25
        self.CATCH_RADIUS = 30
        self.MAX_STEPS = 2500
        self.POINTS_TO_WIN_GAME = 3

        self.TILE_GRID_DIMS = (16, 10) # columns, rows
        self.tile_w = self.width / self.TILE_GRID_DIMS[0]
        self.tile_h = self.height / self.TILE_GRID_DIMS[1]

        # --- State Variables ---
        self.player_pos = pygame.math.Vector2(0, 0)
        self.opponent_pos = pygame.math.Vector2(0, 0)
        self.ball_pos = pygame.math.Vector2(0, 0)
        self.ball_vel = pygame.math.Vector2(0, 0)
        
        self.player_has_ball = False
        self.gravity_direction = 1  # 1 for down, -1 for up
        self.last_move_direction = pygame.math.Vector2(1, 0)
        
        self.player_score = 0
        self.opponent_score = 0
        self.player_game_wins = 0
        self.opponent_game_wins = 0
        self.base_opponent_speed = 2.0
        self.opponent_speed = self.base_opponent_speed

        self.tile_grid = np.zeros(self.TILE_GRID_DIMS, dtype=int)
        
        self.active_powerups = {}
        self.gravity_well = None # (pos, timer)
        self.particles = []
        self.starfield = []

        self.steps = 0
        self.game_over = False
        self.last_space_press = False # For one-shot actions

        self._create_starfield()
        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is for debugging, not needed in production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_score = 0
        self.opponent_score = 0
        self.player_game_wins = 0
        self.opponent_game_wins = 0
        self.opponent_speed = self.base_opponent_speed
        self.steps = 0
        self.game_over = False
        
        self._reset_round_state(starter='player')
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        # --- Action Processing ---
        movement, space_action, shift_action = action[0], action[1] == 1, action[2] == 1
        
        # Player movement and shooting direction
        move_vector = pygame.math.Vector2(0, 0)
        if movement == 1: move_vector.y = -1
        elif movement == 2: move_vector.y = 1
        elif movement == 3: move_vector.x = -1
        elif movement == 4: move_vector.x = 1
        
        if move_vector.length() > 0:
            self.last_move_direction = move_vector.normalize()
        
        current_speed = self.PLAYER_SPEED
        if 'speed' in self.active_powerups:
            current_speed *= 1.5
        
        self.player_pos += move_vector * current_speed
        self._clamp_player_position()
        
        # Tile matching check
        if move_vector.length() > 0:
            reward += self._check_tile_match()

        # Space action (Shoot/Catch) - trigger on press, not hold
        if space_action and not self.last_space_press:
            if self.player_has_ball:
                # Shoot
                self.player_has_ball = False
                self.ball_vel = self.last_move_direction * self.BALL_MAX_SPEED
                # sfx: shoot_ball
            else:
                # Catch
                if self.player_pos.distance_to(self.ball_pos) < self.CATCH_RADIUS:
                    self.player_has_ball = True
                    self.ball_vel = pygame.math.Vector2(0, 0)
                    reward += 1.0 # Reward for catching
                    # sfx: catch_ball
                else:
                    reward -= 0.1 # Penalty for missing a catch
        self.last_space_press = space_action

        # Shift action (Gravity Flip)
        if shift_action:
            self.gravity_direction *= -1
            # sfx: gravity_flip

        # --- Game Logic Update ---
        self._update_powerups()
        self._update_ball_physics()
        self._update_opponent_ai()
        
        # Moving towards ball reward
        if not self.player_has_ball and move_vector.length() > 0:
            dist_before = (self.player_pos - move_vector * current_speed).distance_to(self.ball_pos)
            dist_after = self.player_pos.distance_to(self.ball_pos)
            if dist_after < dist_before:
                reward += 0.01

        # --- Scoring and Round/Game Reset ---
        terminated = False
        if self.ball_pos.x > self.width:
            self.player_score += 1
            reward += 5.0 # Point score reward
            # sfx: score_goal
            self._create_particles(self.player_pos, self.PLAYER_COLOR, 50)
            if self.player_score >= self.POINTS_TO_WIN_GAME:
                self.player_game_wins += 1
                reward += 20.0 # Game win reward
                self.opponent_speed += 0.25 # Difficulty increase
                if self.player_game_wins >= 2:
                    reward += 50.0 # Match win reward
                    terminated = True
                else:
                    self._reset_game()
            else:
                self._reset_round_state(starter='opponent')
        elif self.ball_pos.x < 0:
            self.opponent_score += 1
            reward -= 5.0 # Point loss penalty
            # sfx: opponent_scores
            self._create_particles(self.opponent_pos, self.OPPONENT_COLOR, 50)
            if self.opponent_score >= self.POINTS_TO_WIN_GAME:
                self.opponent_game_wins += 1
                reward -= 20.0 # Game loss penalty
                if self.opponent_game_wins >= 2:
                    reward -= 50.0 # Match loss penalty
                    terminated = True
                else:
                    self._reset_game()
            else:
                self._reset_round_state(starter='player')
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # In gym, truncated implies terminated
        
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _reset_round_state(self, starter='player'):
        """Resets positions for a new point, but keeps scores."""
        self.gravity_direction = 1
        self.active_powerups.clear()
        self.gravity_well = None

        if starter == 'player':
            self.player_pos = pygame.math.Vector2(self.width * 0.25, self.height / 2)
            self.opponent_pos = pygame.math.Vector2(self.width * 0.75, self.height / 2)
            self.player_has_ball = True
        else: # Opponent starts
            self.player_pos = pygame.math.Vector2(self.width * 0.25, self.height / 2)
            self.opponent_pos = pygame.math.Vector2(self.width * 0.75, self.height / 2)
            self.player_has_ball = False
            self.ball_pos = self.opponent_pos.copy()
            # Opponent serves immediately
            self.ball_vel = pygame.math.Vector2(-1, self.np_random.uniform(-0.5, 0.5)).normalize() * self.BALL_MAX_SPEED

        self.ball_vel = pygame.math.Vector2(0, 0)
        self._update_ball_physics() # To snap ball to player if they have it
        self._fill_tile_grid()

    def _reset_game(self):
        """Resets after a game is won, resetting scores for the new game."""
        self.player_score = 0
        self.opponent_score = 0
        self._reset_round_state(starter='player')

    def _fill_tile_grid(self):
        self.tile_grid = self.np_random.integers(0, len(self.TILE_COLORS), size=self.TILE_GRID_DIMS)

    def _clamp_player_position(self):
        self.player_pos.x = max(self.PLAYER_SIZE, min(self.player_pos.x, self.width - self.PLAYER_SIZE))
        self.player_pos.y = max(self.PLAYER_SIZE, min(self.player_pos.y, self.height - self.PLAYER_SIZE))

    def _update_ball_physics(self):
        if self.player_has_ball:
            self.ball_pos = self.player_pos + self.last_move_direction * (self.PLAYER_SIZE + self.BALL_SIZE + 2)
            self.ball_vel = pygame.math.Vector2(0, 0)
            return

        # Apply gravity well
        if self.gravity_well:
            well_pos, _ = self.gravity_well
            to_well = well_pos - self.ball_pos
            if to_well.length() > 0:
                # Inverse square law for attraction
                force = to_well.normalize() * 500 / max(1, to_well.length_squared())
                self.ball_vel += force
        
        # Apply global gravity
        self.ball_vel.y += self.GRAVITY_STRENGTH * self.gravity_direction
        
        # Clamp ball speed
        if self.ball_vel.length() > self.BALL_MAX_SPEED:
            self.ball_vel.scale_to_length(self.BALL_MAX_SPEED)
            
        self.ball_pos += self.ball_vel
        
        # Wall bounces
        if self.ball_pos.y < self.BALL_SIZE or self.ball_pos.y > self.height - self.BALL_SIZE:
            self.ball_pos.y = max(self.BALL_SIZE, min(self.ball_pos.y, self.height - self.BALL_SIZE))
            self.ball_vel.y *= -0.8 # Dampen bounce
            # sfx: bounce

    def _update_opponent_ai(self):
        target_pos = pygame.math.Vector2(0, 0)
        
        if self.player_has_ball:
            # Defensive: try to get between player and goal
            target_pos.x = self.width * 0.75
            target_pos.y = self.player_pos.y
        else:
            # Offensive: go for the ball
            target_pos = self.ball_pos

        move_dir = target_pos - self.opponent_pos
        if move_dir.length() > self.opponent_speed:
            move_dir.scale_to_length(self.opponent_speed)
        
        self.opponent_pos += move_dir
        self.opponent_pos.x = max(self.width / 2, min(self.opponent_pos.x, self.width - self.PLAYER_SIZE))
        self.opponent_pos.y = max(self.PLAYER_SIZE, min(self.opponent_pos.y, self.height - self.PLAYER_SIZE))

        # Opponent catches ball
        if not self.player_has_ball and self.opponent_pos.distance_to(self.ball_pos) < self.CATCH_RADIUS:
            # Opponent shoots immediately towards player's goal
            self.ball_vel = pygame.math.Vector2(-1, self.np_random.uniform(-0.5, 0.5)).normalize() * self.BALL_MAX_SPEED
            # sfx: opponent_shoot

    def _check_tile_match(self):
        """Checks for tile matches under the player and awards powerups."""
        px, py = self.player_pos
        tile_x = int(px / self.tile_w)
        tile_y = int(py / self.tile_h)
        
        if not (0 <= tile_x < self.TILE_GRID_DIMS[0] and 0 <= tile_y < self.TILE_GRID_DIMS[1]):
            return 0

        color_to_match = self.tile_grid[tile_x, tile_y]
        
        # Use BFS to find all connected tiles of the same color
        q = deque([(tile_x, tile_y)])
        visited = set([(tile_x, tile_y)])
        match_group = []

        while q:
            x, y = q.popleft()
            match_group.append((x, y))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.TILE_GRID_DIMS[0] and 0 <= ny < self.TILE_GRID_DIMS[1] \
                   and (nx, ny) not in visited and self.tile_grid[nx, ny] == color_to_match:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        
        if len(match_group) >= 3:
            # sfx: tile_match
            for x, y in match_group:
                self.tile_grid[x, y] = self.np_random.integers(0, len(self.TILE_COLORS)) # Refill
            
            if color_to_match == 0: # Blue -> Speed
                self.active_powerups['speed'] = 150 # 5 seconds at 30fps
            elif color_to_match == 1: # Red -> Gravity Well
                self.gravity_well = (self.player_pos.copy(), 150)
            
            return 2.0 # Reward for matching
        return 0
    
    def _update_powerups(self):
        # Speed boost
        if 'speed' in self.active_powerups:
            self.active_powerups['speed'] -= 1
            if self.active_powerups['speed'] <= 0:
                del self.active_powerups['speed']
        
        # Gravity well
        if self.gravity_well:
            pos, timer = self.gravity_well
            timer -= 1
            if timer <= 0:
                self.gravity_well = None
            else:
                self.gravity_well = (pos, timer)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_tiles()
        self._render_court_lines()
        self._render_effects()
        self._render_entities()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.player_score,
            "opponent_score": self.opponent_score,
            "game_wins": self.player_game_wins,
            "opponent_game_wins": self.opponent_game_wins,
            "steps": self.steps,
        }

    def _create_starfield(self):
        self.starfield = []
        for i in range(200):
            # z represents depth: 1 is close, 3 is far
            z = random.choice([1, 2, 3])
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            color_val = 150 // z
            color = (color_val, color_val, color_val + 50)
            self.starfield.append([x, y, z, color])

    def _render_background(self):
        for star in self.starfield:
            star[0] = (star[0] - 1 / star[2]) % self.width # Parallax scroll
            pygame.gfxdraw.pixel(self.screen, int(star[0]), int(star[1]), star[3])

    def _render_tiles(self):
        for x in range(self.TILE_GRID_DIMS[0]):
            for y in range(self.TILE_GRID_DIMS[1]):
                color_index = self.tile_grid[x, y]
                base_color = self.TILE_COLORS[color_index]
                rect = pygame.Rect(x * self.tile_w, y * self.tile_h, self.tile_w, self.tile_h)
                pygame.draw.rect(self.screen, base_color, rect, 1)

    def _render_court_lines(self):
        line_color = (255, 255, 255, 50)
        pygame.draw.line(self.screen, line_color, (self.width / 2, 0), (self.width / 2, self.height), 2)
        pygame.draw.circle(self.screen, line_color, (self.width / 2, self.height / 2), 50, 2)

    def _render_entities(self):
        # Opponent
        pygame.gfxdraw.filled_circle(self.screen, int(self.opponent_pos.x), int(self.opponent_pos.y), self.PLAYER_SIZE, self.OPPONENT_COLOR)
        pygame.gfxdraw.aacircle(self.screen, int(self.opponent_pos.x), int(self.opponent_pos.y), self.PLAYER_SIZE, self.OPPONENT_COLOR)

        # Player
        glow_color = (*self.PLAYER_COLOR, 50)
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_SIZE + 5, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_SIZE, self.PLAYER_COLOR)
        pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_SIZE, self.PLAYER_COLOR)

        # Ball
        if not self.player_has_ball:
            glow_color = (*self.BALL_COLOR, 100)
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_SIZE + 3, glow_color)
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_SIZE, self.BALL_COLOR)
            pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_SIZE, self.BALL_COLOR)
    
    def _render_effects(self):
        # Particles
        for p in self.particles[:]:
            p[0] += p[1] # pos += vel
            p[3] -= 1 # lifetime
            if p[3] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(p[3] * 5)))
                color = (*p[2], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p[0].x), int(p[0].y), int(p[3]/10), color)
        
        # Speed boost trail
        if 'speed' in self.active_powerups:
            alpha = min(255, self.active_powerups['speed'] * 2)
            color = (*self.POWERUP_COLORS['speed'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_SIZE, color)

        # Gravity Well
        if self.gravity_well:
            pos, timer = self.gravity_well
            radius = 30 + 10 * math.sin(self.steps * 0.2)
            alpha = min(255, timer * 2)
            color = (*self.POWERUP_COLORS['gravity'], alpha)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), color)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius)+1, color)
    
    def _render_ui(self):
        # Score
        score_text = f"{self.player_score} - {self.opponent_score}"
        text_surf = self.font_large.render(score_text, True, (255, 255, 255))
        self.screen.blit(text_surf, (self.width / 2 - text_surf.get_width() / 2, 10))
        
        # Game Wins
        wins_text = f"GAMES: {self.player_game_wins} - {self.opponent_game_wins}"
        text_surf = self.font_small.render(wins_text, True, (200, 200, 200))
        self.screen.blit(text_surf, (self.width - text_surf.get_width() - 10, 10))

        # Gravity Indicator
        grav_text = "GRAV: " + ("DOWN" if self.gravity_direction == 1 else "UP")
        color = (200, 200, 255) if self.gravity_direction == 1 else (255, 200, 200)
        text_surf = self.font_small.render(grav_text, True, color)
        self.screen.blit(text_surf, (10, self.height - text_surf.get_height() - 10))
        
        # Powerup Status
        y_offset = 10
        if 'speed' in self.active_powerups:
            text_surf = self.font_small.render("SPEED BOOST", True, self.POWERUP_COLORS['speed'])
            self.screen.blit(text_surf, (10, y_offset))
            y_offset += 20
        if self.gravity_well:
            text_surf = self.font_small.render("GRAV WELL", True, self.POWERUP_COLORS['gravity'])
            self.screen.blit(text_surf, (10, y_offset))

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            vel = pygame.math.Vector2(random.uniform(-3, 3), random.uniform(-3, 3))
            lifetime = random.randint(20, 50)
            self.particles.append([pos.copy(), vel, color, lifetime])

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Example of how to run the environment ---
    # The following code is for local testing and visualization.
    # It will not be run by the evaluation server.
    
    # Un-comment the line below to run with a display
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # Use arrow keys for movement, Space to shoot/catch, Left Shift to flip gravity
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Nebula Netball")
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # Construct action from keyboard input
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

    print(f"Game Over. Final Info: {info}")
    env.close()