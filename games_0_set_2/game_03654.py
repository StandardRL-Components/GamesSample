
# Generated: 2025-08-28T00:00:25.261973
# Source Brief: brief_03654.md
# Brief Index: 3654

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move. Stay out of the expanding shadow and reach the white shelter to win."
    )

    game_description = (
        "A grid-based survival game. Evade a relentless, procedurally generated shadow and reach the safety of the shelter before you run out of moves or the darkness consumes you."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.CELL_SIZE
        self.MAX_STEPS = 1500
        self.SHADOW_VERTICES_COUNT = 16 # More vertices for a rounder look

        # --- Colors ---
        self.COLOR_BG = (26, 28, 44)
        self.COLOR_GRID = (42, 44, 60)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_SHELTER = (255, 255, 255)
        self.COLOR_SHADOW = (0, 0, 0)
        self.COLOR_TEXT = (160, 160, 255)
        self.COLOR_PARTICLE = (200, 200, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        # --- Game State Variables (initialized in reset) ---
        self.np_random = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""
        self.player_pos = [0, 0]
        self.shelter_pos = (0, 0)
        self.shadow_center = [0, 0]
        self.shadow_angles = []
        self.shadow_radii = []
        self.shadow_expansion_rate = 0.0
        self.particles = []

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            # Ensure np_random is initialized even if seed is None
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""

        # Place shelter (top right)
        self.shelter_pos = (self.GRID_WIDTH - 3, 2)

        # Place player (bottom left quadrant)
        px = self.np_random.integers(2, self.GRID_WIDTH // 2)
        py = self.np_random.integers(self.GRID_HEIGHT // 2, self.GRID_HEIGHT - 2)
        self.player_pos = [px, py]

        # Initialize shadow
        self.shadow_center = [self.GRID_WIDTH / 2, self.GRID_HEIGHT / 2]
        self.shadow_angles = [i * 2 * math.pi / self.SHADOW_VERTICES_COUNT for i in range(self.SHADOW_VERTICES_COUNT)]
        self.shadow_radii = [1.0] * self.SHADOW_VERTICES_COUNT
        self.shadow_expansion_rate = 0.8

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        reward = 0.0

        # --- Store old state for reward calculation ---
        old_player_pos = self.player_pos[:]
        old_dist_to_shelter = self._dist(old_player_pos, self.shelter_pos)
        old_dist_to_shadow = self._dist(old_player_pos, self.shadow_center)

        # --- Update Player ---
        if movement == 1:  # Up
            self.player_pos[1] -= 1
        elif movement == 2:  # Down
            self.player_pos[1] += 1
        elif movement == 3:  # Left
            self.player_pos[0] -= 1
        elif movement == 4:  # Right
            self.player_pos[0] += 1
        
        # Clamp player position to grid
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_WIDTH - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- Update Shadow ---
        self._update_shadow()
        
        # --- Update Particles ---
        self._update_particles()

        # --- Calculate Reward ---
        reward += 0.1  # Survival reward

        new_dist_to_shelter = self._dist(self.player_pos, self.shelter_pos)
        if new_dist_to_shelter < old_dist_to_shelter:
            reward += 0.5  # Moved closer to shelter

        new_dist_to_shadow = self._dist(self.player_pos, self.shadow_center)
        if new_dist_to_shadow < old_dist_to_shadow:
            reward -= 0.2  # Moved closer to shadow center (risky)

        # --- Check Termination ---
        terminated, term_reward, message = self._check_termination()
        reward += term_reward
        self.win_message = message
        
        if terminated:
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_shadow(self):
        # Increase difficulty over time
        if self.steps > 0 and self.steps % 500 == 0:
            self.shadow_expansion_rate += 0.05
            
        # Find vertex index closest to player
        player_angle = math.atan2(
            self.player_pos[1] - self.shadow_center[1],
            self.player_pos[0] - self.shadow_center[0]
        )
        angle_diffs = [abs((angle - player_angle + math.pi) % (2 * math.pi) - math.pi) for angle in self.shadow_angles]
        best_idx = np.argmin(angle_diffs)
        
        # Create a weighted probability distribution to expand vertices
        # High probability for the vertex near the player and its neighbors
        probs = np.full(self.SHADOW_VERTICES_COUNT, 0.1 / (self.SHADOW_VERTICES_COUNT - 3))
        probs[best_idx] = 0.7
        probs[(best_idx - 1) % self.SHADOW_VERTICES_COUNT] = 0.1
        probs[(best_idx + 1) % self.SHADOW_VERTICES_COUNT] = 0.1
        
        idx_to_expand = self.np_random.choice(self.SHADOW_VERTICES_COUNT, p=probs)
        
        # Expand the chosen vertex
        self.shadow_radii[idx_to_expand] += self.shadow_expansion_rate
        
        # Add particle effect for feedback
        # sfx: shadow_growth.wav
        angle = self.shadow_angles[idx_to_expand]
        radius = self.shadow_radii[idx_to_expand]
        px = self.shadow_center[0] + math.cos(angle) * radius
        py = self.shadow_center[1] + math.sin(angle) * radius
        
        for _ in range(5):
            particle_angle = angle + self.np_random.uniform(-0.5, 0.5)
            particle_speed = self.np_random.uniform(0.5, 1.5)
            vel = [math.cos(particle_angle) * particle_speed, math.sin(particle_angle) * particle_speed]
            life = self.np_random.integers(10, 20)
            self.particles.append({"pos": [px * self.CELL_SIZE, py * self.CELL_SIZE], "vel": vel, "life": life})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _check_termination(self):
        # Win condition
        if tuple(self.player_pos) == self.shelter_pos:
            # sfx: win_chime.wav
            return True, 100.0, "SHELTER REACHED"
        
        # Loss condition: caught by shadow
        shadow_verts = self._get_shadow_pixel_vertices()
        player_pixel_pos = self._to_pixels(self.player_pos, center=True)
        if self._is_point_in_polygon(player_pixel_pos, shadow_verts):
            # sfx: player_caught.wav
            return True, -100.0, "CONSUMED"
            
        # Loss condition: out of steps
        if self.steps >= self.MAX_STEPS:
            # sfx: timeout_buzz.wav
            return True, -100.0, "OUT OF TIME"
            
        return False, 0.0, ""

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw shadow
        shadow_verts = self._get_shadow_pixel_vertices()
        if len(shadow_verts) > 2:
            pygame.gfxdraw.aapolygon(self.screen, shadow_verts, self.COLOR_SHADOW)
            pygame.gfxdraw.filled_polygon(self.screen, shadow_verts, self.COLOR_SHADOW)
            
        # Draw shelter
        sx, sy = self._to_pixels(self.shelter_pos)
        pygame.draw.rect(self.screen, self.COLOR_SHELTER, (sx, sy, self.CELL_SIZE, self.CELL_SIZE))

        # Draw particles
        for p in self.particles:
            alpha = p["life"] / 20.0
            color = (*self.COLOR_PARTICLE, int(255 * alpha))
            size = int(3 * alpha)
            if size > 0:
                 pygame.draw.circle(self.screen, self.COLOR_PARTICLE, (int(p["pos"][0]), int(p["pos"][1])), size)

        # Draw player
        px, py = self._to_pixels(self.player_pos, center=True)
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.CELL_SIZE // 2 + 3, (*self.COLOR_PLAYER, 50))
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.CELL_SIZE // 2, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, self.CELL_SIZE // 2, self.COLOR_PLAYER)

    def _render_ui(self):
        steps_left_text = self.font_ui.render(f"MOVES LEFT: {self.MAX_STEPS - self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_left_text, (10, 10))

        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_game_over.render(self.win_message, True, self.COLOR_PLAYER)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "shelter_pos": self.shelter_pos,
            "distance_to_shelter": self._dist(self.player_pos, self.shelter_pos),
        }

    # --- Helper Methods ---
    def _dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _to_pixels(self, grid_pos, center=False):
        x = grid_pos[0] * self.CELL_SIZE
        y = grid_pos[1] * self.CELL_SIZE
        if center:
            x += self.CELL_SIZE // 2
            y += self.CELL_SIZE // 2
        return int(x), int(y)

    def _get_shadow_pixel_vertices(self):
        verts = []
        for i in range(self.SHADOW_VERTICES_COUNT):
            angle = self.shadow_angles[i]
            radius = self.shadow_radii[i]
            px = self.shadow_center[0] + math.cos(angle) * radius
            py = self.shadow_center[1] + math.sin(angle) * radius
            verts.append(self._to_pixels((px, py)))
        return verts
    
    def _is_point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set SDL_VIDEODRIVER to a dummy value to run without a display
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # --- Manual Play Setup ---
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    running = True
    
    action = np.array([0, 0, 0]) # Start with no-op
    
    print("\n" + "="*50)
    print(env.game_description)
    print(env.user_guide)
    print("="*50 + "\n")

    while running:
        terminated = False
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()

                # --- Map keys to actions for one step ---
                current_action = np.array([0, 0, 0]) # Default no-op
                if not env.game_over:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_UP]:
                        current_action[0] = 1
                    elif keys[pygame.K_DOWN]:
                        current_action[0] = 2
                    elif keys[pygame.K_LEFT]:
                        current_action[0] = 3
                    elif keys[pygame.K_RIGHT]:
                        current_action[0] = 4
                
                    obs, reward, terminated, truncated, info = env.step(current_action)
                    print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

        # --- Drawing ---
        # The environment's observation is already a rendered frame
        frame = env._get_observation()
        # Pygame uses (width, height), numpy uses (height, width)
        # The observation is (height, width, 3), so we need to transpose it for pygame
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            # Pause on game over, wait for R to reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        wait_for_reset = False


    env.close()