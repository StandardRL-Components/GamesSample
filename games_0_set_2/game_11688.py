import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:22:14.955529
# Source Brief: brief_01688.md
# Brief Index: 1688
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a bouncing ball to clear a
    board of numbered tiles through chain reactions.

    **Visuals**: Clean, geometric, vibrant style with satisfying particle effects.
    **Gameplay**: Aim a ball, launch it, and watch it bounce off walls and tiles.
    When a tile's number reaches zero, it explodes, triggering adjacent tiles.
    The goal is to clear 75% of the tiles within a 60-second time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Aim and launch a ball to clear a board of numbered tiles. Hitting a tile reduces its number, "
        "and when it reaches zero, it triggers a chain reaction with adjacent tiles."
    )
    user_guide = (
        "Use the arrow keys to aim the launcher and press space to launch the ball."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 3600 # 60 seconds at 60 FPS target
    WIN_PERCENTAGE = 0.75

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 45, 60)
    COLOR_BALL = (0, 255, 255) # Bright Cyan
    COLOR_BALL_GLOW = (0, 150, 150)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TIMER_WARN = (255, 100, 100)
    COLOR_TILE_VALUES = {
        3: (50, 100, 255),  # Blue
        2: (200, 50, 200),  # Magenta
        1: (255, 80, 80),   # Red
    }
    PARTICLE_COLORS = [(255, 255, 100), (255, 200, 50), (255, 150, 0)]

    # Game Grid
    GRID_COLS = 16
    GRID_ROWS = 10
    TILE_SIZE = 32
    GAP_SIZE = 4
    GRID_WIDTH = GRID_COLS * (TILE_SIZE + GAP_SIZE) - GAP_SIZE
    GRID_HEIGHT = GRID_ROWS * (TILE_SIZE + GAP_SIZE) - GAP_SIZE
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 20 # Lower grid for UI space

    # Ball Physics
    BALL_RADIUS = 8
    LAUNCH_SPEED = 5.0
    MIN_BALL_SPEED_TO_RESET = 0.1
    BALL_RESET_TIMER_MAX = 180 # 3 seconds
    ANGLE_ADJUST_RATE = 0.05 # Radians per step

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_tile = pygame.font.SysFont("Arial", 20, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0.0
        
        self.tiles = []
        self.initial_tile_count = 0
        self.cleared_tile_count = 0

        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_state = "READY" # READY or ACTIVE
        self.ball_reset_timer = 0
        
        self.launch_angle = -math.pi / 4
        self.space_pressed_last_frame = False

        self.particles = []

        # Initialize state by calling reset
        # self.reset() # reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cleared_tile_count = 0
        self.initial_tile_count = 0
        
        self.ball_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.GRID_OFFSET_Y - 20)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_state = "READY"
        self.ball_reset_timer = 0

        self.launch_angle = -math.pi / 4
        self.space_pressed_last_frame = False
        
        self.particles.clear()
        
        self._generate_tiles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_this_step = 0.0
        self.steps += 1

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held)

        # --- Update Game Logic ---
        self._update_ball()
        self._update_particles()
        
        # --- Check Termination ---
        terminated = False
        truncated = False
        win = self.initial_tile_count > 0 and (self.cleared_tile_count / self.initial_tile_count) >= self.WIN_PERCENTAGE
        
        if win:
            self.reward_this_step += 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            self.reward_this_step -= 100.0
            terminated = True # Technically truncated, but for this game, time limit is a terminal loss condition
            self.game_over = True

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        clear_percentage = 0
        if self.initial_tile_count > 0:
            clear_percentage = self.cleared_tile_count / self.initial_tile_count
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": (self.MAX_STEPS - self.steps) / 60.0,
            "clear_percentage": clear_percentage
        }

    # --- Game Logic Methods ---

    def _generate_tiles(self):
        self.tiles = []
        self.initial_tile_count = 0
        for r in range(self.GRID_ROWS):
            row = []
            for c in range(self.GRID_COLS):
                # 70% chance of a tile spawning
                if self.np_random.random() < 0.7:
                    value = self.np_random.integers(1, 4)
                    row.append(value)
                    self.initial_tile_count += 1
                else:
                    row.append(0) # 0 means no tile
            self.tiles.append(row)

    def _handle_input(self, movement, space_held):
        if self.ball_state == "READY":
            # Adjust angle
            if movement == 1: self.launch_angle -= self.ANGLE_ADJUST_RATE # Up
            if movement == 2: self.launch_angle += self.ANGLE_ADJUST_RATE # Down
            if movement == 3: self.launch_angle -= self.ANGLE_ADJUST_RATE # Left
            if movement == 4: self.launch_angle += self.ANGLE_ADJUST_RATE # Right
            self.launch_angle = self.launch_angle % (2 * math.pi)

            # Launch ball on space press (rising edge)
            if space_held and not self.space_pressed_last_frame:
                self.ball_state = "ACTIVE"
                self.ball_vel.x = math.cos(self.launch_angle) * self.LAUNCH_SPEED
                self.ball_vel.y = math.sin(self.launch_angle) * self.LAUNCH_SPEED
                # sound: ball_launch.wav
        
        self.space_pressed_last_frame = space_held

    def _update_ball(self):
        if self.ball_state != "ACTIVE":
            return

        self.ball_pos += self.ball_vel
        
        # Ball reset logic
        if self.ball_vel.length() < self.MIN_BALL_SPEED_TO_RESET:
            self.ball_reset_timer += 1
        else:
            self.ball_reset_timer = 0
            
        if self.ball_reset_timer > self.BALL_RESET_TIMER_MAX:
            self._reset_ball()
            return
            
        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS < 0:
            self.ball_pos.x = self.BALL_RADIUS
            self.ball_vel.x *= -1
            # sound: wall_bounce.wav
        elif self.ball_pos.x + self.BALL_RADIUS > self.SCREEN_WIDTH:
            self.ball_pos.x = self.SCREEN_WIDTH - self.BALL_RADIUS
            self.ball_vel.x *= -1
            # sound: wall_bounce.wav
        if self.ball_pos.y - self.BALL_RADIUS < 0:
            self.ball_pos.y = self.BALL_RADIUS
            self.ball_vel.y *= -1
            # sound: wall_bounce.wav
        elif self.ball_pos.y + self.BALL_RADIUS > self.SCREEN_HEIGHT:
            self.ball_pos.y = self.SCREEN_HEIGHT - self.BALL_RADIUS
            self.ball_vel.y *= -1
            # sound: wall_bounce.wav

        # Tile collisions
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.tiles[r][c] > 0:
                    tile_rect = self._get_tile_rect(r, c)
                    if ball_rect.colliderect(tile_rect):
                        self._handle_tile_collision(r, c, tile_rect)
                        return # Handle one collision per frame for simplicity

    def _handle_tile_collision(self, r, c, tile_rect):
        # Determine collision side to reflect velocity correctly
        overlap = self.ball_pos - pygame.Vector2(tile_rect.center)
        dx = abs(overlap.x) - tile_rect.width / 2
        dy = abs(overlap.y) - tile_rect.height / 2

        if abs(dx) > abs(dy):
            self.ball_vel.x *= -1
            self.ball_pos.x += self.ball_vel.x # Push out
        else:
            self.ball_vel.y *= -1
            self.ball_pos.y += self.ball_vel.y # Push out

        self._trigger_tile_hit(r, c, is_chain_reaction=False)
        # sound: tile_hit.wav

    def _trigger_tile_hit(self, r, c, is_chain_reaction):
        if not (0 <= r < self.GRID_ROWS and 0 <= c < self.GRID_COLS):
            return
        if self.tiles[r][c] <= 0:
            return
        
        self.tiles[r][c] -= 1
        self.reward_this_step += 0.1 # Reward for any hit
        
        if self.tiles[r][c] == 0:
            self.cleared_tile_count += 1
            self.score += 1
            
            if is_chain_reaction:
                self.reward_this_step += 5.0 # Higher reward for chain reactions
            else:
                self.reward_this_step += 1.0 # Base reward for clearing a tile
            
            # sound: tile_explode.wav
            self._create_particles(r, c)
            
            # Trigger chain reaction on neighbors
            self._trigger_tile_hit(r + 1, c, True)
            self._trigger_tile_hit(r - 1, c, True)
            self._trigger_tile_hit(r, c + 1, True)
            self._trigger_tile_hit(r, c - 1, True)

    def _reset_ball(self):
        self.ball_state = "READY"
        self.ball_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.GRID_OFFSET_Y - 20)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_reset_timer = 0

    def _create_particles(self, r, c):
        tile_rect = self._get_tile_rect(r, c)
        center = tile_rect.center
        num_particles = self.np_random.integers(15, 25)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(20, 40)
            color = random.choice(self.PARTICLE_COLORS)
            self.particles.append({"pos": pygame.Vector2(center), "vel": vel, "life": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Damping
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    # --- Rendering Methods ---

    def _render_game(self):
        # Render tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.tiles[r][c] > 0:
                    self._render_tile(r, c)
        
        # Render particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"].x), int(p["pos"].y)), int(p["life"] / 10))

        # Render ball and launch indicator
        if self.ball_state == "READY":
            self._render_launch_indicator()
        
        self._render_ball()

    def _render_tile(self, r, c):
        tile_rect = self._get_tile_rect(r, c)
        value = self.tiles[r][c]
        color = self.COLOR_TILE_VALUES.get(value, (100, 100, 100))
        
        pygame.draw.rect(self.screen, color, tile_rect, border_radius=4)
        
        text_surf = self.font_tile.render(str(value), True, self.COLOR_BG)
        text_rect = text_surf.get_rect(center=tile_rect.center)
        self.screen.blit(text_surf, text_rect)

    def _render_ball(self):
        # Glow effect
        glow_radius = self.BALL_RADIUS * 2.5
        glow_alpha = 30
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_BALL_GLOW, glow_alpha), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (int(self.ball_pos.x - glow_radius), int(self.ball_pos.y - glow_radius)))
        
        # Main ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_launch_indicator(self):
        length = 50
        end_pos_x = self.ball_pos.x + math.cos(self.launch_angle) * length
        end_pos_y = self.ball_pos.y + math.sin(self.launch_angle) * length
        pygame.draw.aaline(self.screen, self.COLOR_BALL, self.ball_pos, (end_pos_x, end_pos_y), 2)

    def _render_ui(self):
        # Cleared percentage
        clear_percentage = 0
        if self.initial_tile_count > 0:
            clear_percentage = self.cleared_tile_count / self.initial_tile_count * 100
        
        clear_text = f"Cleared: {clear_percentage:.1f}%"
        clear_surf = self.font_ui.render(clear_text, True, self.COLOR_TEXT)
        self.screen.blit(clear_surf, (10, 10))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 60.0
        timer_color = self.COLOR_TEXT if time_left > 10 else self.COLOR_TIMER_WARN
        timer_text = f"Time: {max(0, time_left):.2f}"
        timer_surf = self.font_ui.render(timer_text, True, timer_color)
        timer_rect = timer_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_surf, timer_rect)

    # --- Helper Methods ---

    def _get_tile_rect(self, r, c):
        x = self.GRID_OFFSET_X + c * (self.TILE_SIZE + self.GAP_SIZE)
        y = self.GRID_OFFSET_Y + r * (self.TILE_SIZE + self.GAP_SIZE)
        return pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
        
    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == "__main__":
    # The main loop is for human interaction and visualization.
    # The core environment remains headless.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    # Use a separate window for human play
    human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Chain Reaction Bouncer")
    
    obs, info = env.reset()
    done = False
    
    # --- Human Controls Mapping ---
    # This is for manual testing and does not affect the agent's action space.
    # The agent still must use the MultiDiscrete([5, 2, 2]) space.
    movement_action = 0  # 0: none, 1: up, 2: down, 3: left, 4: right
    space_action = 0     # 0: released, 1: held
    shift_action = 0     # 0: released, 1: held
    
    print("\n--- Human Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset Environment")
    print("Q: Quit")
    print("----------------------\n")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                
        # Map keyboard state to agent action space
        keys = pygame.key.get_pressed()
        
        # Movement: For human play, we'll let multiple keys combine,
        # but the agent action space only allows one direction at a time.
        # We'll prioritize the last one in this list.
        movement_action = 0
        if keys[pygame.K_UP]: movement_action = 1
        if keys[pygame.K_DOWN]: movement_action = 2
        if keys[pygame.K_LEFT]: movement_action = 3
        if keys[pygame.K_RIGHT]: movement_action = 4
            
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement_action, space_action, shift_action]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Cleared: {info['clear_percentage']:.2%}")
        
        if terminated or truncated:
            print("--- Episode Finished ---")
            print(f"Final Score: {info['score']}")
            print(f"Final Clear Percentage: {info['clear_percentage']:.2%}")
            print("Resetting environment...")
            obs, info = env.reset()
            pygame.time.wait(2000)

        # Render the observation to the human-controlled window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(60) # Limit to 60 FPS for human play

    env.close()