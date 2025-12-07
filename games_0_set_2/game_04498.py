
# Generated: 2025-08-28T02:34:47.957809
# Source Brief: brief_04498.md
# Brief Index: 4498

        
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
        "Controls: Arrow keys to move your hunter. Collect the colorful bugs."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hunt colorful, shimmering bugs in a procedurally generated garden before you run out of time. Avoid the grey blocks!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    WIN_CONDITION = 50
    MAX_STEPS = 1800  # 60 seconds at 30 FPS
    NUM_OBSTACLES = 10
    NUM_BUGS = 50

    PLAYER_RADIUS = 10
    PLAYER_SPEED = 4
    BUG_RADIUS = 6
    OBSTACLE_SIZE_MIN = 20
    OBSTACLE_SIZE_MAX = 40

    # Colors
    COLOR_BG_DARK = (25, 60, 25)
    COLOR_BG_LIGHT = (35, 80, 35)
    COLOR_WALL = (50, 30, 20)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_PLAYER_OUTLINE = (255, 200, 200)
    COLOR_OBSTACLE = (80, 80, 90)
    COLOR_OBSTACLE_OUTLINE = (60, 60, 70)
    COLOR_TEXT = (255, 255, 255)
    BUG_COLORS = [
        (100, 255, 100), (100, 100, 255), (255, 100, 255),
        (255, 255, 100), (100, 255, 255), (255, 150, 50)
    ]


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.bugs = None
        self.obstacles = None
        self.steps = None
        self.score = None
        self.bugs_collected = None
        self.game_over = None
        self.last_dist_to_bug = None

        self.np_random = None # Will be seeded in reset()

        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.bugs_collected = 0
        self.game_over = False

        # Center player
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)

        # Procedurally generate level
        self.obstacles = self._spawn_obstacles()
        self.bugs = self._spawn_bugs()

        self.last_dist_to_bug = self._get_nearest_bug_dist()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30) # Maintain 30 FPS

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        reward = 0
        terminated = False

        # --- Update game logic ---
        self._handle_movement(movement)
        
        # Calculate distance-based reward
        if self.last_dist_to_bug is not None:
            current_dist = self._get_nearest_bug_dist()
            if current_dist < self.last_dist_to_bug:
                reward += 0.1  # Closer to a bug
            else:
                reward -= 0.02 # Further from a bug
            self.last_dist_to_bug = current_dist

        # Check for bug collection
        collected_bug = self._check_bug_collisions()
        if collected_bug:
            reward += 1.0
            self.bugs_collected += 1
            # Recalculate nearest bug for next step's reward
            self.last_dist_to_bug = self._get_nearest_bug_dist()
            # SFX: play("collect.wav")

        # Check for obstacle collision
        if self._check_obstacle_collisions():
            reward -= 10.0
            self.game_over = True
            terminated = True
            # SFX: play("crash.wav")

        # Check for win/loss conditions
        if self.bugs_collected >= self.WIN_CONDITION:
            reward += 100.0
            self.game_over = True
            terminated = True
            # SFX: play("win.wav")

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            reward -= 10.0
            self.game_over = True
            terminated = True
            # SFX: play("timeout.wav")
        
        self.score += reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen with background
        self._render_background()
        self._render_walls()

        # Render all game elements
        self._render_obstacles()
        self._render_bugs()
        self._render_player()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bugs_collected": self.bugs_collected,
        }

    # --- Helper and Rendering Methods ---

    def _handle_movement(self, movement):
        if movement == 1:  # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED

        # Boundary checks
        wall_thickness = 5
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS + wall_thickness, self.WIDTH - self.PLAYER_RADIUS - wall_thickness)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS + wall_thickness, self.HEIGHT - self.PLAYER_RADIUS - wall_thickness)

    def _spawn_obstacles(self):
        obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            for _ in range(100): # Max 100 attempts to find a valid spot
                size = self.np_random.integers(self.OBSTACLE_SIZE_MIN, self.OBSTACLE_SIZE_MAX, 2)
                pos = self.np_random.integers(10, [self.WIDTH - 10 - size[0], self.HEIGHT - 10 - size[1]])
                new_obstacle = pygame.Rect(pos[0], pos[1], size[0], size[1])

                # Check for overlap with player start and other obstacles
                if new_obstacle.colliderect(pygame.Rect(self.player_pos[0]-50, self.player_pos[1]-50, 100, 100)):
                    continue
                
                if any(new_obstacle.colliderect(obs) for obs in obstacles):
                    continue

                obstacles.append(new_obstacle)
                break
        return obstacles

    def _spawn_bugs(self):
        bugs = []
        for _ in range(self.NUM_BUGS):
            for _ in range(100): # Max 100 attempts
                pos = self.np_random.uniform(
                    [self.BUG_RADIUS + 5, self.BUG_RADIUS + 5],
                    [self.WIDTH - self.BUG_RADIUS - 5, self.HEIGHT - self.BUG_RADIUS - 5]
                )
                
                # Check overlap with obstacles
                collides_obstacle = any(obs.collidepoint(pos) for obs in self.obstacles)
                if collides_obstacle:
                    continue

                # Check overlap with other bugs
                collides_bug = any(np.linalg.norm(pos - bug['pos']) < 2 * self.BUG_RADIUS for bug in bugs)
                if collides_bug:
                    continue
                
                bugs.append({
                    'pos': pos,
                    'color': random.choice(self.BUG_COLORS)
                })
                break
        return bugs

    def _check_bug_collisions(self):
        collected_any = False
        for i in range(len(self.bugs) - 1, -1, -1):
            bug = self.bugs[i]
            dist = np.linalg.norm(self.player_pos - bug['pos'])
            if dist < self.PLAYER_RADIUS + self.BUG_RADIUS:
                self.bugs.pop(i)
                collected_any = True
        return collected_any

    def _check_obstacle_collisions(self):
        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_RADIUS,
            self.player_pos[1] - self.PLAYER_RADIUS,
            self.PLAYER_RADIUS * 2,
            self.PLAYER_RADIUS * 2
        )
        for obs in self.obstacles:
            if obs.colliderect(player_rect):
                # More accurate circle-rect collision
                closest_x = np.clip(self.player_pos[0], obs.left, obs.right)
                closest_y = np.clip(self.player_pos[1], obs.top, obs.bottom)
                dist = math.hypot(self.player_pos[0] - closest_x, self.player_pos[1] - closest_y)
                if dist < self.PLAYER_RADIUS:
                    return True
        return False

    def _get_nearest_bug_dist(self):
        if not self.bugs:
            return 0
        
        bug_positions = np.array([bug['pos'] for bug in self.bugs])
        distances = np.linalg.norm(bug_positions - self.player_pos, axis=1)
        return np.min(distances)

    def _render_background(self):
        tile_size = 40
        for y in range(0, self.HEIGHT, tile_size):
            for x in range(0, self.WIDTH, tile_size):
                rect = (x, y, tile_size, tile_size)
                color = self.COLOR_BG_LIGHT if (x // tile_size + y // tile_size) % 2 == 0 else self.COLOR_BG_DARK
                pygame.draw.rect(self.screen, color, rect)

    def _render_walls(self):
        wall_thickness = 5
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, wall_thickness))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, self.HEIGHT - wall_thickness, self.WIDTH, wall_thickness))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, wall_thickness, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - wall_thickness, 0, wall_thickness, self.HEIGHT))

    def _render_obstacles(self):
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, obs, 2)

    def _render_bugs(self):
        # Pulsing animation for bugs
        pulse = (math.sin(self.steps * 0.2) + 1) / 2  # Varies between 0 and 1
        current_radius = int(self.BUG_RADIUS * (1 + pulse * 0.2))

        for bug in self.bugs:
            pos = (int(bug['pos'][0]), int(bug['pos'][1]))
            color = bug['color']
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], current_radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], current_radius, color)

    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        # Glow effect
        glow_radius = int(self.PLAYER_RADIUS * 1.5)
        glow_color = (*self.COLOR_PLAYER, 50) # RGBA with low alpha
        temp_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (pos[0]-glow_radius, pos[1]-glow_radius))
        
        # Player body
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        # Bugs Collected
        bugs_text = f"BUGS: {self.bugs_collected} / {self.WIN_CONDITION}"
        bugs_surf = self.font_main.render(bugs_text, True, self.COLOR_TEXT)
        self.screen.blit(bugs_surf, (self.WIDTH // 2 - bugs_surf.get_width() // 2, self.HEIGHT - 30))

        # Time remaining
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = f"TIME: {time_left}"
        time_surf = self.font_main.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 15, 10))

        # Score (RL reward)
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 10))

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space (requires reset to be run first)
        # We can't fully test this here, as reset needs to be called by the user first.
        # But we can check the class attributes.
        assert self.observation_space.shape == (self.HEIGHT, self.WIDTH, 3)
        assert self.observation_space.dtype == np.uint8
        
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Bug Hunt")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    print(env.game_description)
    print(env.user_guide)

    while running:
        movement = 0 # No-op by default
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        # The action space requires all 3 components
        action = [movement, 0, 0] # space and shift are not used

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # Pygame uses a different coordinate system for surfaces, so we need to transpose
        # the observation from (H, W, C) to (W, H, C) and then rotate/flip.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Bugs Collected: {info['bugs_collected']}")
            # Wait a moment, then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

    env.close()