import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T19:52:28.702993
# Source Brief: brief_03221.md
# Brief Index: 3221
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a tilt-based maze game. The player controls a
    rolling ball, navigating a maze to collect checkpoints against a time limit.
    Halfway through, the ball becomes magnetized, attracting remaining checkpoints.
    The game prioritizes visual quality and responsive, momentum-based physics.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a rolling ball by tilting the maze. Collect all checkpoints and reach the exit "
        "before time runs out."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to tilt the board and guide the ball through the maze."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 30 * self.FPS  # 30-second time limit

        # --- Colors (Vibrant & High Contrast) ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_WALL = (100, 110, 120)
        self.COLOR_START = (40, 100, 40)
        self.COLOR_EXIT = (100, 40, 40)
        self.COLOR_BALL = (0, 150, 255)
        self.COLOR_BALL_MAGNET = (150, 50, 255)
        self.COLOR_CHECKPOINT = (255, 215, 0) # Gold
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_BG = (0, 0, 0, 128)

        # --- Physics & Game Parameters ---
        self.BALL_RADIUS = 12
        self.TILT_FORCE = 0.3
        self.FRICTION = 0.985
        self.MAX_SPEED = 8.0
        self.WALL_BOUNCE_DAMPING = 0.7
        self.CHECKPOINT_RADIUS = 8
        self.MAGNET_STRENGTH = 150.0  # Proportional to 1/r for a soft pull
        self.MAGNET_ACTIVATION_RATIO = 0.5

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24, bold=True)

        # --- Maze Definition ---
        self._define_maze()
        
        # --- Initialize State Variables (properly set in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.checkpoints = []
        self.total_checkpoints = 0
        self.is_magnetized = False
        self.particles = []
        self.last_dist_to_target = 0.0

    def _define_maze(self):
        """Defines the static layout of the maze, start/exit zones, and checkpoints."""
        self.start_zone = pygame.Rect(20, 20, 60, 60)
        self.exit_zone = pygame.Rect(self.WIDTH - 80, self.HEIGHT - 80, 60, 60)

        wall_data = [
            (0, 0, self.WIDTH, 5), (0, 0, 5, self.HEIGHT), (0, self.HEIGHT - 5, self.WIDTH, 5), (self.WIDTH - 5, 0, 5, self.HEIGHT), # Borders
            (100, 0, 10, 150), (100, 200, 10, 200), (200, 50, 10, 250),
            (100, 145, 110, 10), (0, 250, 150, 10), (200, 295, 200, 10),
            (300, 0, 10, 200), (400, 100, 10, 300), (400, 95, 150, 10),
            (500, 95, 10, 150), (400, 295, 150, 10), (self.WIDTH - 150, self.HEIGHT - 150, 150, 10),
        ]
        self.walls = [pygame.Rect(r) for r in wall_data]

        self.checkpoint_data = [
            (155, 50), (155, 250), (55, 325), (255, 250), (255, 50), 
            (355, 250), (455, 50), (555, 175), (455, 350), (self.WIDTH - 50, self.HEIGHT - 50)
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.ball_pos = pygame.Vector2(self.start_zone.center)
        self.ball_vel = pygame.Vector2(0, 0)
        
        self.checkpoints = [pygame.Vector2(p) for p in self.checkpoint_data]
        self.total_checkpoints = len(self.checkpoints)
        self.is_magnetized = False
        self.particles = []

        self.last_dist_to_target = self._get_dist_to_next_target()
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0

        # 1. Apply Forces
        acceleration = pygame.Vector2(0, 0)
        if movement == 1: acceleration.y -= self.TILT_FORCE  # Up
        elif movement == 2: acceleration.y += self.TILT_FORCE  # Down
        elif movement == 3: acceleration.x -= self.TILT_FORCE  # Left
        elif movement == 4: acceleration.x += self.TILT_FORCE  # Right

        if self.is_magnetized and self.checkpoints:
            for cp in self.checkpoints:
                vec_to_cp = cp - self.ball_pos
                dist = vec_to_cp.length()
                if dist > self.BALL_RADIUS:
                    force_magnitude = self.MAGNET_STRENGTH / max(dist, self.BALL_RADIUS)
                    acceleration += vec_to_cp.normalize() * force_magnitude

        # 2. Update Physics
        self.ball_vel += acceleration
        self.ball_vel *= self.FRICTION
        if self.ball_vel.length() > self.MAX_SPEED:
            self.ball_vel.scale_to_length(self.MAX_SPEED)
        self.ball_pos += self.ball_vel

        # 3. Handle Wall Collisions
        for wall in self.walls:
            closest_x = max(wall.left, min(self.ball_pos.x, wall.right))
            closest_y = max(wall.top, min(self.ball_pos.y, wall.bottom))
            distance = self.ball_pos.distance_to((closest_x, closest_y))

            if distance < self.BALL_RADIUS:
                # sfx: wall_thud.wav
                reward -= 0.01
                overlap = self.BALL_RADIUS - distance
                collision_normal = self.ball_pos - pygame.Vector2(closest_x, closest_y)
                if collision_normal.length() > 0:
                    collision_normal.normalize_ip()
                
                self.ball_pos += collision_normal * overlap
                self.ball_vel = self.ball_vel.reflect(collision_normal) * self.WALL_BOUNCE_DAMPING

        # 4. Handle Checkpoint Collection
        collected_this_step = [i for i, cp in enumerate(self.checkpoints) if self.ball_pos.distance_to(cp) < self.BALL_RADIUS + self.CHECKPOINT_RADIUS]
        if collected_this_step:
            # sfx: checkpoint_collect.wav
            reward += 1.0 * len(collected_this_step)
            self.score += len(collected_this_step)
            for i in sorted(collected_this_step, reverse=True):
                self.checkpoints.pop(i)

        if not self.is_magnetized and (self.total_checkpoints - len(self.checkpoints)) >= self.total_checkpoints * self.MAGNET_ACTIVATION_RATIO:
            self.is_magnetized = True
            # sfx: magnet_activate.wav

        # 5. Continuous Reward for approaching next target
        new_dist = self._get_dist_to_next_target()
        reward += (self.last_dist_to_target - new_dist) * 0.1
        self.last_dist_to_target = new_dist

        # 6. Update Visual Effects
        self._update_particles()

        # 7. Check Termination Conditions
        self.steps += 1
        terminated = False
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 100.0
            # sfx: game_lose.wav

        if not self.checkpoints and self.exit_zone.collidepoint(self.ball_pos):
            terminated = True
            reward += 100.0
            self.score += 10 # Bonus for finishing
            # sfx: game_win.wav
        
        if terminated:
            self.game_over = True

        obs = self._get_observation()
        info = self._get_info()
        truncated = False # This game does not have a truncation condition separate from termination
        return obs, reward, terminated, truncated, info

    def _get_dist_to_next_target(self):
        """Calculates distance to the closest checkpoint, or the exit if none are left."""
        if not self.checkpoints:
            return self.ball_pos.distance_to(self.exit_zone.center)
        
        return min(self.ball_pos.distance_to(cp) for cp in self.checkpoints)

    def _update_particles(self):
        """Adds new particles and updates/removes old ones for the ball's trail."""
        p_color = self.COLOR_BALL_MAGNET if self.is_magnetized else self.COLOR_BALL
        p_vel = self.ball_vel * -0.1 + pygame.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        self.particles.append({'pos': self.ball_pos.copy(), 'vel': p_vel, 'radius': self.BALL_RADIUS * 0.5, 'color': p_color, 'lifespan': 20})

        for p in self.particles:
            p['pos'] += p['vel']
            p['radius'] *= 0.95
            p['lifespan'] -= 1
        
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0.5]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders all primary game elements."""
        pygame.draw.rect(self.screen, self.COLOR_START, self.start_zone)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_zone)

        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 20))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

        for cp in self.checkpoints:
            self._render_glow_circle(cp, self.CHECKPOINT_RADIUS, self.COLOR_CHECKPOINT, 1.8)
        
        ball_color = self.COLOR_BALL_MAGNET if self.is_magnetized else self.COLOR_BALL
        self._render_glow_circle(self.ball_pos, self.BALL_RADIUS, ball_color, 2.0)

    def _render_glow_circle(self, pos, radius, color, glow_factor):
        """Renders a circle with a soft glowing effect using layered transparent circles."""
        pos_int = (int(pos.x), int(pos.y))
        glow_radius = int(radius * glow_factor)
        
        for i in range(glow_radius, int(radius), -1):
            alpha = int(50 * (1 - (i - radius) / (glow_radius - radius))**2)
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], i, glow_color)

        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(radius), color)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(radius), color)

    def _render_ui(self):
        """Renders the UI text for time and checkpoint count."""
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"TIME: {time_left:.2f}"
        checkpoints_text = f"LEFT: {len(self.checkpoints)}/{self.total_checkpoints}"

        self._render_text_with_bg(time_text, (10, 10))
        self._render_text_with_bg(checkpoints_text, (10, 35))

    def _render_text_with_bg(self, text, position):
        """Renders text with a semi-transparent background for readability."""
        text_surface = self.font_ui.render(text, True, self.COLOR_UI_TEXT)
        text_rect = text_surface.get_rect(topleft=position)
        bg_rect = text_rect.inflate(10, 6)
        bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surface.fill(self.COLOR_UI_BG)
        self.screen.blit(bg_surface, bg_rect)
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "checkpoints_left": len(self.checkpoints),
            "is_magnetized": self.is_magnetized,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Verifies that the environment conforms to the specified Gymnasium interface."""
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing and demonstration.
    
    # The environment must be instantiated with render_mode='rgb_array' for headless operation
    # but we can create a window for manual play.
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit() # a quit/init cycle is needed to change video driver
    pygame.init()
    pygame.font.init()

    env = GameEnv()
    obs, info = env.reset()
    
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tilt Maze")
    
    terminated = False
    total_reward = 0.0
    
    keys_held = {k: False for k in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]}

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Objective: Collect all gold checkpoints and reach the red exit zone in 30 seconds.")

    while not terminated:
        action = [0, 0, 0] # Default action: no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key in keys_held:
                keys_held[event.key] = True
            if event.type == pygame.KEYUP and event.key in keys_held:
                keys_held[event.key] = False

        if keys_held[pygame.K_UP]: action[0] = 1
        elif keys_held[pygame.K_DOWN]: action[0] = 2
        elif keys_held[pygame.K_LEFT]: action[0] = 3
        elif keys_held[pygame.K_RIGHT]: action[0] = 4

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # The observation is the rendered frame. Transpose it back for display.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        env.screen.blit(surf, (0, 0))

        pygame.display.flip()
        env.clock.tick(env.FPS)

    print(f"\nGame Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()