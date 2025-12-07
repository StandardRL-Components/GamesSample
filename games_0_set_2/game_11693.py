import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:26:21.175679
# Source Brief: brief_01693.md
# Brief Index: 1693
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple

# Define a named tuple for branch data for better readability
Branch = namedtuple('Branch', ['start', 'end', 'angle', 'length', 'depth', 'id'])

class GameEnv(gym.Env):
    """
    Fractal Grower Gymnasium Environment

    The player controls a cursor to select a growth point on a recursive fractal.
    The goal is to grow the fractal for 60 seconds without causing collisions.

    Action Space: MultiDiscrete([5, 2, 2])
    - 0: Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - 1: Place Branch (0: released, 1: pressed)
    - 2: Shift (unused)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +0.1 per step (survival)
    - +1.0 for a successful branch placement
    - -5.0 for a collision
    - +50.0 for surviving the full duration
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Grow a beautiful fractal by strategically placing new branches. Race against the clock and avoid "
        "self-collisions to maximize your creation's complexity."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press space to grow a new branch from the nearest endpoint."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 1800  # Corresponds to 60 seconds at 30 FPS
        self.FPS = 30

        self.CURSOR_SPEED = 8
        self.INITIAL_COLLISIONS = 5
        
        self.BRANCH_ANGLE_DEG = 35
        self.BRANCH_LENGTH_SCALE = 1.08
        self.MAX_BRANCH_LENGTH = 150
        self.MIN_BRANCH_LENGTH = 2

        # --- Colors ---
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_COLLISION = (255, 50, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_SHADOW = (10, 10, 20)
        self.COLOR_TIMER_BAR = (70, 120, 255)

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
            self.font_ui = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 24, bold=True)
            self.font_ui_small = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 18)
        except pygame.error:
            self.font_ui = pygame.font.SysFont("sans-serif", 24, bold=True)
            self.font_ui_small = pygame.font.SysFont("sans-serif", 18)

        # --- Game State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.collisions_remaining = 0
        self.cursor_pos = [0, 0]
        self.branches = []
        self.particles = []
        self.collision_flashes = []
        self.prev_space_held = False
        self.next_branch_id = 0
        self.background_stars = []

        # self.reset() is not called in __init__ as per Gymnasium standard practice
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.collisions_remaining = self.INITIAL_COLLISIONS
        
        self.cursor_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
        
        self.branches = []
        self.particles = []
        self.collision_flashes = []
        self.prev_space_held = False
        self.next_branch_id = 0

        # Create the initial root branch
        root_length = 40
        root_start = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT)
        root_end = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - root_length)
        root_branch = Branch(start=root_start, end=root_end, angle=-90, length=root_length, depth=0, id=self._get_next_branch_id())
        self.branches.append(root_branch)

        # Initialize background stars for parallax effect
        self.background_stars = []
        for _ in range(150):
            self.background_stars.append({
                "pos": [random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)],
                "size": random.uniform(0.5, 1.5),
                "speed": random.uniform(0.1, 0.3)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Base reward for survival

        # --- Handle Input ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._move_cursor(movement)

        # On the rising edge of the space button press
        if space_held and not self.prev_space_held:
            # Sfx: place_attempt.wav
            placement_result, data = self._place_branch()
            if placement_result == 'success':
                # Sfx: success.wav
                reward += 1.0
                new_branches = data
                self.branches.extend(new_branches)
                self.score += len(new_branches)
                # Spawn success particles
                spawn_point = new_branches[0].start
                for _ in range(15):
                    self._spawn_particle(spawn_point, (200, 255, 200), 20)
            elif placement_result == 'collision':
                # Sfx: collision.wav
                reward -= 5.0
                self.collisions_remaining -= 1
                collision_point = data
                self.collision_flashes.append({"pos": collision_point, "life": 15})
                # Spawn collision particles
                for _ in range(25):
                    self._spawn_particle(collision_point, self.COLOR_COLLISION, 30)

        self.prev_space_held = space_held

        # --- Update Game State ---
        self._update_animations()

        # --- Check Termination ---
        terminated = self.collisions_remaining <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.steps >= self.MAX_STEPS:
                # Sfx: victory.wav
                reward += 50.0  # Victory bonus
            else:
                # Sfx: game_over.wav
                pass

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_next_branch_id(self):
        id = self.next_branch_id
        self.next_branch_id += 1
        return id

    def _move_cursor(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2:  # Down
            self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3:  # Left
            self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4:  # Right
            self.cursor_pos[0] += self.CURSOR_SPEED
        
        # Clamp cursor to screen bounds
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

    def _place_branch(self):
        if not self.branches:
            return 'failure', None

        # Find the closest branch endpoint to the cursor
        min_dist = float('inf')
        parent_branch = None
        for branch in self.branches:
            dist = math.hypot(self.cursor_pos[0] - branch.end[0], self.cursor_pos[1] - branch.end[1])
            if dist < min_dist:
                min_dist = dist
                parent_branch = branch
        
        if parent_branch is None:
            return 'failure', None

        # Create two new potential child branches
        new_branches = []
        angles = [parent_branch.angle + self.BRANCH_ANGLE_DEG, parent_branch.angle - self.BRANCH_ANGLE_DEG]
        new_length = min(self.MAX_BRANCH_LENGTH, parent_branch.length * self.BRANCH_LENGTH_SCALE)
        
        if new_length < self.MIN_BRANCH_LENGTH:
            return 'failure', None # Stop growth if branches are too small

        for angle in angles:
            rad = math.radians(angle)
            end_x = parent_branch.end[0] + new_length * math.cos(rad)
            end_y = parent_branch.end[1] + new_length * math.sin(rad)
            new_branch = Branch(
                start=parent_branch.end, end=(end_x, end_y), angle=angle,
                length=new_length, depth=parent_branch.depth + 1, id=self._get_next_branch_id()
            )
            new_branches.append(new_branch)

        # Check for collisions
        for new_b in new_branches:
            for existing_b in self.branches:
                # Don't check against the direct parent
                if existing_b.id == parent_branch.id:
                    continue
                
                # Use a helper to check for line segment intersection
                if self._line_intersection(new_b.start, new_b.end, existing_b.start, existing_b.end):
                    # For simplicity, we get the midpoint of the new branch as collision point
                    collision_point = ((new_b.start[0] + new_b.end[0]) / 2, (new_b.start[1] + new_b.end[1]) / 2)
                    return 'collision', collision_point
        
        return 'success', new_branches

    def _update_animations(self):
        # Update particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

        # Update collision flashes
        for f in self.collision_flashes:
            f['life'] -= 1
        self.collision_flashes = [f for f in self.collision_flashes if f['life'] > 0]

        # Update background stars
        for star in self.background_stars:
            star['pos'][1] += star['speed']
            if star['pos'][1] > self.SCREEN_HEIGHT:
                star['pos'][1] = 0
                star['pos'][0] = random.uniform(0, self.SCREEN_WIDTH)


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_fractal()
        self._render_particles_and_flashes()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.background_stars:
            color_val = int(star['speed'] * 100 + 50)
            color = (color_val, color_val, color_val + 20)
            pygame.gfxdraw.pixel(self.screen, int(star['pos'][0]), int(star['pos'][1]), color)

    def _render_fractal(self):
        for branch in self.branches:
            hue = (branch.depth * 25) % 360
            color = pygame.Color(0)
            color.hsla = (hue, 100, 60, 100)
            
            width = max(1, int(8 / (branch.depth + 1)))
            
            # Using pygame.draw.line with width for thickness
            pygame.draw.line(self.screen, color, branch.start, branch.end, width)

    def _render_particles_and_flashes(self):
        # Render particles
        for p in self.particles:
            alpha = p['life'] / p['max_life']
            radius = int(p['size'] * alpha)
            if radius > 0:
                color = (*p['color'], int(alpha * 255))
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (int(p['pos'][0] - radius), int(p['pos'][1] - radius)))

        # Render collision flashes
        for f in self.collision_flashes:
            alpha = f['life'] / 15.0
            radius = int(40 * (1.0 - alpha))
            color = (*self.COLOR_COLLISION, int(alpha * 150))
            
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            self.screen.blit(temp_surf, (int(f['pos'][0] - radius), int(f['pos'][1] - radius)))

    def _render_cursor(self):
        x, y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        # Pulsing glow effect
        pulse = (math.sin(self.steps * 0.2) + 1) / 2  # Varies between 0 and 1
        
        # Outer glow
        glow_radius = int(12 + pulse * 4)
        glow_alpha = int(50 + pulse * 20)
        pygame.gfxdraw.filled_circle(self.screen, x, y, glow_radius, (*self.COLOR_CURSOR, glow_alpha))
        pygame.gfxdraw.aacircle(self.screen, x, y, glow_radius, (*self.COLOR_CURSOR, glow_alpha))
        
        # Inner circle
        pygame.gfxdraw.filled_circle(self.screen, x, y, 5, self.COLOR_CURSOR)
        pygame.gfxdraw.aacircle(self.screen, x, y, 5, self.COLOR_CURSOR)
        
    def _render_ui(self):
        # Helper to draw text with a shadow
        def draw_text(text, font, color, pos):
            shadow_surface = font.render(text, True, self.COLOR_UI_SHADOW)
            self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
            text_surface = font.render(text, True, color)
            self.screen.blit(text_surface, pos)

        # Score / Complexity
        score_text = f"COMPLEXITY: {self.score}"
        draw_text(score_text, self.font_ui, self.COLOR_UI_TEXT, (15, 10))

        # Collisions / Integrity
        integrity_text = "INTEGRITY: "
        integrity_pos_x, integrity_pos_y = 15, 40
        draw_text(integrity_text, self.font_ui_small, self.COLOR_UI_TEXT, (integrity_pos_x, integrity_pos_y))
        
        heart_size = 15
        heart_spacing = 20
        start_x = integrity_pos_x + self.font_ui_small.size(integrity_text)[0]
        for i in range(self.collisions_remaining):
            color = (200, 80, 80)
            p1 = (start_x + i * heart_spacing, integrity_pos_y + heart_size // 2)
            p2 = (start_x + i * heart_spacing + heart_size // 2, integrity_pos_y + heart_size)
            p3 = (start_x + i * heart_spacing + heart_size, integrity_pos_y + heart_size // 2)
            p4 = (start_x + i * heart_spacing + heart_size // 2, integrity_pos_y)
            pygame.draw.polygon(self.screen, color, [p1,p4,p3,p2])


        # Timer Bar
        timer_width = 200
        timer_height = 15
        timer_x = self.SCREEN_WIDTH - timer_width - 15
        timer_y = 15
        
        time_fraction = max(0, (self.MAX_STEPS - self.steps) / self.MAX_STEPS)
        
        # Bar background
        pygame.draw.rect(self.screen, self.COLOR_UI_SHADOW, (timer_x, timer_y, timer_width, timer_height), border_radius=4)
        # Bar foreground
        bar_color = self.COLOR_TIMER_BAR
        if time_fraction < 0.2: # Turns red when time is low
            bar_color = (255, 80, 80)
        pygame.draw.rect(self.screen, bar_color, (timer_x, timer_y, int(timer_width * time_fraction), timer_height), border_radius=4)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "collisions_remaining": self.collisions_remaining,
        }

    def _spawn_particle(self, pos, color, max_life):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.5, 2.5)
        self.particles.append({
            'pos': list(pos),
            'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
            'life': random.randint(max_life // 2, max_life),
            'max_life': max_life,
            'size': random.uniform(2, 5),
            'color': color
        })

    def _line_intersection(self, p1, p2, p3, p4):
        # Using vector cross product method to check for intersection
        # Add a small epsilon to avoid floating point issues and touching-point collisions
        epsilon = 1e-6
        
        # Convert points to numpy arrays for vector operations
        P1, P2, P3, P4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)
        
        r = P2 - P1
        s = P4 - P3
        
        r_cross_s = np.cross(r, s)
        q_minus_p = P3 - P1
        
        if abs(r_cross_s) < epsilon: # Parallel or collinear
            return False # For simplicity, ignore collinear case

        t = np.cross(q_minus_p, s) / r_cross_s
        u = np.cross(q_minus_p, r) / r_cross_s

        # Check if intersection point is within both line segments (but not at the endpoints)
        return (epsilon < t < 1 - epsilon) and (epsilon < u < 1 - epsilon)

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
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # --- Manual Play Example ---
    # The __main__ block is for local testing and visualization.
    # It requires a graphical display.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Fractal Grower")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    total_reward = 0
    
    # Game loop for manual control
    while not done:
        movement = 0  # No movement
        space_held = 0 # Not pressed
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        action = [movement, space_held, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Cap the frame rate
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score (Complexity): {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()