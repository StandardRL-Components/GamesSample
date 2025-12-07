import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:42:57.522394
# Source Brief: brief_02465.md
# Brief Index: 2465
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls the tilt of a maze
    to guide a marble to an exit, avoiding black holes and shifting obstacles.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Tilt the maze to guide a marble to the exit. Navigate around shifting walls and avoid "
        "falling into black holes before time runs out."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to tilt the maze and guide the marble."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 20  # As specified in the brief
    MAX_STEPS = 1200  # 60 seconds * 20 FPS

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_WALL = (180, 190, 200)
    COLOR_WALL_SHADOW = (100, 110, 120)
    COLOR_EXIT = (40, 220, 120)
    COLOR_EXIT_SHADOW = (20, 150, 80)
    COLOR_OBSTACLE = (80, 90, 100)
    COLOR_OBSTACLE_SHADOW = (50, 60, 70)
    COLOR_PLAYER = (230, 50, 50)
    COLOR_PLAYER_HIGHLIGHT = (255, 150, 150)
    COLOR_PLAYER_SHADOW = (0, 0, 0, 90) # RGBA for transparency
    COLOR_BLACK_HOLE_CENTER = (0, 0, 0)
    COLOR_BLACK_HOLE_EDGE = (30, 30, 40)
    COLOR_UI_TEXT = (240, 240, 240)

    # --- Physics ---
    TILT_FORCE = 0.08
    FRICTION = 0.985
    MAX_VELOCITY = 6.0
    MARBLE_RADIUS = 10
    WALL_THICKNESS = 10

    # --- Obstacle Mechanics ---
    OBSTACLE_SHIFT_INTERVAL = 100 # 5 seconds at 20 FPS
    OBSTACLE_SHIFT_DISTANCE = 10
    OBSTACLE_SPEED = 0.2 # pixels per frame to achieve shift in 50 frames


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24)
            self.font_msg = pygame.font.SysFont("Impact", 60)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 30)
            self.font_msg = pygame.font.SysFont(None, 74)

        # --- State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_won = False
        self.timer = 0.0
        self.marble_pos = pygame.Vector2(0, 0)
        self.marble_vel = pygame.Vector2(0, 0)
        self.maze_layout = []
        self.black_holes = []
        self.obstacles = []
        self.exit_rect = None

        # self.reset() # reset() is called by the wrapper/user
        # self.validate_implementation() # Validation should not be in init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_won = False
        self.timer = self.MAX_STEPS / self.FPS

        self.marble_pos = pygame.Vector2(50, 50)
        self.marble_vel = pygame.Vector2(0, 0)

        self._create_level()

        return self._get_observation(), self._get_info()

    def _create_level(self):
        """Initializes the positions of all game elements for a new episode."""
        WT = self.WALL_THICKNESS
        W = self.SCREEN_WIDTH
        H = self.SCREEN_HEIGHT

        # Define maze walls
        self.maze_layout = [
            pygame.Rect(0, 0, W, WT),
            pygame.Rect(0, H - WT, W, H),
            pygame.Rect(0, 0, WT, H),
            pygame.Rect(W - WT, 0, WT, H),
            pygame.Rect(100, 0, WT, 150),
            pygame.Rect(100, 250, WT, 150),
            pygame.Rect(200, 100, 200, WT),
            pygame.Rect(300, 200, WT, 150),
            pygame.Rect(450, 0, WT, 300),
        ]

        # Define black holes (position, radius)
        self.black_holes = [
            {'pos': pygame.Vector2(150, 200), 'radius': 15},
            {'pos': pygame.Vector2(400, 350), 'radius': 20},
            {'pos': pygame.Vector2(550, 80), 'radius': 12},
        ]

        # Define shifting obstacles
        self.obstacles = [
            {'rect': pygame.Rect(200, 280, 80, WT), 'start_x': 200, 'dir': 1},
            {'rect': pygame.Rect(500, 180, 80, WT), 'start_x': 500, 'dir': -1},
        ]

        # Define exit
        self.exit_rect = pygame.Rect(W - 60, H - 60, 50, 50)


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # --- Update Time and State ---
        self.steps += 1
        self.timer = max(0, (self.MAX_STEPS - self.steps) / self.FPS)

        # --- Process Action ---
        movement = action[0]
        tilt = pygame.Vector2(0, 0)
        if movement == 1: # Up
            tilt.y = -self.TILT_FORCE
        elif movement == 2: # Down
            tilt.y = self.TILT_FORCE
        elif movement == 3: # Left
            tilt.x = -self.TILT_FORCE
        elif movement == 4: # Right
            tilt.x = self.TILT_FORCE
        # space_held (action[1]) and shift_held (action[2]) are unused

        # --- Apply Physics and Collisions ---
        self._update_obstacles()
        self._apply_physics_and_collisions(tilt)

        # --- Check Termination Conditions ---
        terminated = False
        reward = 0.1  # Survival reward

        # 1. Reached Exit
        if self.exit_rect.collidepoint(self.marble_pos):
            terminated = True
            self.game_won = True
            reward = 100.0
            # sfx: victory_sound.play()

        # 2. Fell into Black Hole
        for hole in self.black_holes:
            if self.marble_pos.distance_to(hole['pos']) < hole['radius']:
                terminated = True
                reward = -10.0
                # sfx: black_hole_sound.play()
                break

        # 3. Ran out of time
        if self.steps >= self.MAX_STEPS:
            terminated = True
            if not self.game_won:
                reward = -10.0

        self.game_over = terminated
        self.score += reward

        # Truncated is always False for this environment
        truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_obstacles(self):
        """Updates the position of shifting obstacles."""
        for obs in self.obstacles:
            target_x = obs['start_x']
            if (self.steps // self.OBSTACLE_SHIFT_INTERVAL) % 2 != 0:
                target_x += self.OBSTACLE_SHIFT_DISTANCE * obs['dir']

            # Move towards target
            if abs(obs['rect'].x - target_x) > self.OBSTACLE_SPEED:
                move_dir = 1 if target_x > obs['rect'].x else -1
                obs['rect'].x += self.OBSTACLE_SPEED * move_dir


    def _apply_physics_and_collisions(self, tilt):
        """Updates marble velocity and position, and handles all collisions."""
        # Apply tilt force and friction
        self.marble_vel += tilt
        self.marble_vel *= self.FRICTION

        # Clamp velocity
        if self.marble_vel.length() > self.MAX_VELOCITY:
            self.marble_vel.scale_to_length(self.MAX_VELOCITY)

        # Update position
        self.marble_pos += self.marble_vel

        # --- Collision Detection and Response ---
        marble_rect = pygame.Rect(
            self.marble_pos.x - self.MARBLE_RADIUS,
            self.marble_pos.y - self.MARBLE_RADIUS,
            self.MARBLE_RADIUS * 2,
            self.MARBLE_RADIUS * 2
        )

        all_walls = self.maze_layout + [obs['rect'] for obs in self.obstacles]

        for wall in all_walls:
            if marble_rect.colliderect(wall):
                # sfx: bounce_sound.play()
                # Find overlap to determine collision side
                dx = self.marble_pos.x - wall.centerx
                dy = self.marble_pos.y - wall.centery
                w_half, h_half = wall.width / 2, wall.height / 2
                overlap_x = w_half + self.MARBLE_RADIUS - abs(dx)
                overlap_y = h_half + self.MARBLE_RADIUS - abs(dy)

                if overlap_x > 0 and overlap_y > 0:
                    if overlap_x < overlap_y: # Horizontal collision
                        self.marble_vel.x *= -0.8 # Lose some energy on bounce
                        self.marble_pos.x += overlap_x if dx > 0 else -overlap_x
                    else: # Vertical collision
                        self.marble_vel.y *= -0.8
                        self.marble_pos.y += overlap_y if dy > 0 else -overlap_y
                    # Re-clamp velocity after bounce
                    if self.marble_vel.length() > self.MAX_VELOCITY:
                        self.marble_vel.scale_to_length(self.MAX_VELOCITY)


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders all primary game elements."""
        shadow_offset = 3

        # Render Exit
        exit_shadow_rect = self.exit_rect.move(shadow_offset, shadow_offset)
        pygame.draw.rect(self.screen, self.COLOR_EXIT_SHADOW, exit_shadow_rect)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect)

        # Render Walls
        for wall in self.maze_layout:
            wall_shadow = wall.move(shadow_offset, shadow_offset)
            pygame.draw.rect(self.screen, self.COLOR_WALL_SHADOW, wall_shadow)
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        # Render Obstacles
        for obs in self.obstacles:
            obs_shadow = obs['rect'].move(shadow_offset, shadow_offset)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_SHADOW, obs_shadow)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'])

        # Render Black Holes
        for hole in self.black_holes:
            pos = (int(hole['pos'].x), int(hole['pos'].y))
            radius = int(hole['radius'])
            for i in range(radius, 0, -2):
                alpha = 1 - (i / radius)
                color = self.COLOR_BLACK_HOLE_EDGE
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, self.COLOR_BLACK_HOLE_CENTER)

        # Render Marble
        self._render_marble()

    def _render_marble(self):
        """Renders the marble with shadow and highlight for a 3D effect."""
        pos = (int(self.marble_pos.x), int(self.marble_pos.y))

        # Shadow
        shadow_pos = (pos[0] + 3, pos[1] + 3)
        shadow_surface = pygame.Surface((self.MARBLE_RADIUS*2, self.MARBLE_RADIUS*2), pygame.SRCALPHA)
        pygame.draw.circle(shadow_surface, self.COLOR_PLAYER_SHADOW, (self.MARBLE_RADIUS, self.MARBLE_RADIUS), self.MARBLE_RADIUS)
        self.screen.blit(shadow_surface, (shadow_pos[0] - self.MARBLE_RADIUS, shadow_pos[1] - self.MARBLE_RADIUS))

        # Main sphere
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.MARBLE_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.MARBLE_RADIUS, self.COLOR_PLAYER)

        # Highlight
        highlight_offset = pygame.Vector2(-1, -1).normalize() * self.MARBLE_RADIUS * 0.4
        highlight_pos = (int(pos[0] + highlight_offset.x), int(pos[1] + highlight_offset.y))
        pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], int(self.MARBLE_RADIUS * 0.5), self.COLOR_PLAYER_HIGHLIGHT)

    def _render_ui(self):
        """Renders the UI text elements."""
        # Timer
        timer_text = f"Time: {self.timer:.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (15, 15))

        # Score
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 15, 15))

        # Game Over / Win Message
        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_EXIT if self.game_won else self.COLOR_PLAYER
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
        }

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
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Example ---
    # This block will not run in the hosted environment but is useful for local testing.
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Setup a display window for manual play
    pygame.display.set_caption("Marble Maze - Manual Control")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    total_reward = 0.0
    
    print("--- Controls ---")
    print("Arrow Keys: Tilt Maze")
    print("R: Reset Environment")
    print("Q: Quit")

    while not done:
        action = [0, 0, 0] # Default action: no-op, no buttons
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0.0
                    print("--- Environment Reset ---")

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

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            print(f"Episode Finished! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(GameEnv.FPS)

    env.close()