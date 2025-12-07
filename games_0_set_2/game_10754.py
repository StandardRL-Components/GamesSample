import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:59:55.268560
# Source Brief: brief_00754.md
# Brief Index: 754
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls the diameter of three pipe
    segments to optimize fluid flow. The goal is to maximize the amount of fluid
    (represented by particles) that passes through the system within a time limit.

    **Visuals:**
    - A clean, technical aesthetic with a dark background and grid.
    - Fluid is visualized as glowing particles.
    - Particle color indicates fluid viscosity (blue=low, red=high).
    - Pipes change diameter visually in response to actions.

    **Gameplay:**
    - The agent adjusts the diameter of three connected pipe segments.
    - Wider pipes decrease viscosity, allowing particles to flow faster.
    - Narrower pipes increase viscosity, slowing particles down.
    - Score is awarded for each particle that successfully exits the system.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control the diameter of three pipe segments to optimize fluid flow. "
        "Maximize the number of particles that pass through the system before time runs out."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to adjust the first pipe's diameter and ←→ for the second. "
        "Use Space and Shift to adjust the third pipe."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds game
        self.WIN_SCORE = 1000

        self.DIAMETER_MIN = 10
        self.DIAMETER_MAX = 50
        self.DIAMETER_CHANGE_RATE = 1.0

        self.PARTICLE_SPAWN_RATE = 4  # Particles to spawn per step
        self.PARTICLE_BASE_SPEED = 0.05  # Progress per step at viscosity 1
        self.PARTICLE_SIZE = 4

        # --- Colors ---
        self.COLORS = {
            "bg": (20, 30, 40),
            "grid": (30, 45, 60),
            "pipe": (120, 130, 140),
            "pipe_border": (80, 90, 100),
            "text": (220, 230, 240),
            "text_shadow": (10, 15, 20),
            "visc_low": (0, 150, 255),
            "visc_high": (255, 50, 50),
            "win": (100, 255, 100),
            "lose": (255, 100, 100),
            "flash": (255, 255, 255),
        }

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_end = pygame.font.Font(None, 72)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.pipe_segments = []
        
        # --- Pipe Path Definition ---
        self.pipe_nodes = [
            pygame.math.Vector2(50, 100),
            pygame.math.Vector2(250, 100),
            pygame.math.Vector2(400, 250),
            pygame.math.Vector2(590, 250),
        ]

        # self.reset() is called by the wrapper/runner, no need to call it here.
        # self.validate_implementation() # This is for debugging, can be removed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles.clear()

        # Initialize pipe segments
        self.pipe_segments.clear()
        for i in range(len(self.pipe_nodes) - 1):
            start_node = self.pipe_nodes[i]
            end_node = self.pipe_nodes[i+1]
            path_vector = end_node - start_node
            
            self.pipe_segments.append({
                "start": start_node,
                "end": end_node,
                "vector": path_vector,
                "length": path_vector.length(),
                "normal": path_vector.normalize().rotate(90),
                "diameter": (self.DIAMETER_MIN + self.DIAMETER_MAX) / 2,
                "viscosity": 5.0,
                "action_flash": 0, # Countdown for visual feedback
            })
        
        self._update_pipe_viscosities()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1

        # 1. Handle Actions
        self._handle_actions(action)

        # 2. Update Game State
        self._update_pipe_viscosities()
        self._spawn_particles()
        finished_particles = self._update_particles()
        
        self.score += finished_particles
        # Sound effect placeholder: play a soft 'blip' for each point scored

        # 3. Calculate Reward
        reward = self._calculate_reward(finished_particles)

        # 4. Check Termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                self.win = True
                reward += 100.0 # Goal-oriented reward
                # Sound effect placeholder: play a 'victory fanfare'
            else:
                # Sound effect placeholder: play a 'failure' sound
                pass

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Action mapping:
        # Movement up/down -> Segment 0
        # Movement left/right -> Segment 1
        # Space/Shift -> Segment 2

        adjustments = [0.0, 0.0, 0.0]
        if movement == 1: adjustments[0] += self.DIAMETER_CHANGE_RATE # Up
        if movement == 2: adjustments[0] -= self.DIAMETER_CHANGE_RATE # Down
        if movement == 4: adjustments[1] += self.DIAMETER_CHANGE_RATE # Right
        if movement == 3: adjustments[1] -= self.DIAMETER_CHANGE_RATE # Left
        if space_held:    adjustments[2] += self.DIAMETER_CHANGE_RATE
        if shift_held:    adjustments[2] -= self.DIAMETER_CHANGE_RATE
        
        for i, adj in enumerate(adjustments):
            if adj != 0:
                seg = self.pipe_segments[i]
                seg["diameter"] = np.clip(
                    seg["diameter"] + adj, self.DIAMETER_MIN, self.DIAMETER_MAX
                )
                seg["action_flash"] = 5 # Flash for 5 frames
                # Sound effect placeholder: play a 'click' or 'whir' sound

    def _update_pipe_viscosities(self):
        for seg in self.pipe_segments:
            # Inverse relationship: max diameter -> min viscosity (1), min diameter -> max viscosity (10)
            normalized_diameter = (seg["diameter"] - self.DIAMETER_MIN) / (self.DIAMETER_MAX - self.DIAMETER_MIN)
            seg["viscosity"] = 10.0 - 9.0 * normalized_diameter
            if seg["action_flash"] > 0:
                seg["action_flash"] -= 1

    def _spawn_particles(self):
        for _ in range(self.PARTICLE_SPAWN_RATE):
            self.particles.append({
                "segment_idx": 0,
                "progress": self.np_random.uniform(0, 0.01), # Stagger start
                "offset": self.np_random.uniform(-0.8, 0.8), # Position within pipe width
            })

    def _update_particles(self):
        finished_count = 0
        
        # Iterate backwards to allow safe removal
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            seg = self.pipe_segments[p["segment_idx"]]

            # Speed is inversely proportional to viscosity
            speed = self.PARTICLE_BASE_SPEED / max(1.0, seg["viscosity"])
            p["progress"] += speed

            if p["progress"] >= 1.0:
                p["segment_idx"] += 1
                if p["segment_idx"] >= len(self.pipe_segments):
                    # Particle reached the end
                    finished_count += 1
                    self.particles.pop(i)
                else:
                    # Move to next segment
                    p["progress"] -= 1.0
        
        return finished_count

    def _calculate_reward(self, finished_particles):
        # Continuous reward for flow rate
        return finished_particles * 0.1

    def _check_termination(self):
        return self.steps >= self.MAX_STEPS or self.score >= self.WIN_SCORE

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS,
            "viscosities": [seg["viscosity"] for seg in self.pipe_segments]
        }

    def _get_observation(self):
        self._render_background()
        self._render_pipes()
        self._render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_end_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLORS["bg"])
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLORS["grid"], (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLORS["grid"], (0, y), (self.WIDTH, y))

    def _render_pipes(self):
        for seg in self.pipe_segments:
            # Main pipe body
            pygame.draw.line(
                self.screen,
                self.COLORS["pipe"],
                (int(seg["start"].x), int(seg["start"].y)),
                (int(seg["end"].x), int(seg["end"].y)),
                int(seg["diameter"])
            )
            # Border for definition
            pygame.draw.line(
                self.screen,
                self.COLORS["pipe_border"],
                (int(seg["start"].x), int(seg["start"].y)),
                (int(seg["end"].x), int(seg["end"].y)),
                int(seg["diameter"]) + 4,
            )
            # Re-draw inner part to cover border overlap at joints
            pygame.draw.line(
                self.screen,
                self.COLORS["pipe"],
                (int(seg["start"].x), int(seg["start"].y)),
                (int(seg["end"].x), int(seg["end"].y)),
                int(seg["diameter"])
            )
            # Action flash effect
            if seg["action_flash"] > 0:
                flash_alpha = int(100 * (seg["action_flash"] / 5.0))
                s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(
                    s,
                    self.COLORS["flash"] + (flash_alpha,),
                    (int(seg["start"].x), int(seg["start"].y)),
                    (int(seg["end"].x), int(seg["end"].y)),
                    int(seg["diameter"]) + 6
                )
                self.screen.blit(s, (0,0))


    def _render_particles(self):
        for p in self.particles:
            seg = self.pipe_segments[p["segment_idx"]]
            
            # Interpolate position along the segment vector
            pos = seg["start"] + p["progress"] * seg["vector"]
            # Add perpendicular offset
            pos += p["offset"] * seg["normal"] * (seg["diameter"] / 2.5)

            # Color based on viscosity
            visc_ratio = (seg["viscosity"] - 1.0) / 9.0
            color = self._lerp_color(self.COLORS["visc_low"], self.COLORS["visc_high"], visc_ratio)
            
            # Glow effect
            glow_color = color + (40,) # Add alpha
            glow_radius = int(self.PARTICLE_SIZE * 2.5)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (int(pos.x - glow_radius), int(pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

            # Core particle
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), self.PARTICLE_SIZE, color)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.PARTICLE_SIZE, color)

    def _render_ui(self):
        # Score
        self._draw_text(f"SCORE: {self.score}", (20, 20), self.font_ui)
        
        # Time
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        self._draw_text(f"TIME: {time_left:.1f}s", (self.WIDTH - 150, 20), self.font_ui)
        
        # Viscosity meters
        self._draw_text("VISCOSITY", (self.WIDTH / 2 - 50, self.HEIGHT - 60), self.font_ui)
        bar_width = 100
        bar_height = 20
        start_x = self.WIDTH / 2 - (1.5 * bar_width + 10)
        
        for i, seg in enumerate(self.pipe_segments):
            x = start_x + i * (bar_width + 10)
            y = self.HEIGHT - 40
            
            visc_ratio = (seg["viscosity"] - 1.0) / 9.0
            color = self._lerp_color(self.COLORS["visc_low"], self.COLORS["visc_high"], visc_ratio)

            # Bar background
            pygame.draw.rect(self.screen, self.COLORS["grid"], (x, y, bar_width, bar_height))
            # Bar fill
            pygame.draw.rect(self.screen, color, (x, y, int(bar_width * visc_ratio), bar_height))
            # Bar border
            pygame.draw.rect(self.screen, self.COLORS["text"], (x, y, bar_width, bar_height), 1)

    def _render_end_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if self.win:
            message = "FLOW OPTIMIZED"
            color = self.COLORS["win"]
        else:
            message = "TIME EXPIRED"
            color = self.COLORS["lose"]
            
        self._draw_text(message, (self.WIDTH/2, self.HEIGHT/2), self.font_end, color=color, center=True)

    def _draw_text(self, text, pos, font, color=None, center=False):
        if color is None:
            color = self.COLORS["text"]
            
        # Draw shadow first
        text_surface = font.render(text, True, self.COLORS["text_shadow"])
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = (pos[0] + 2, pos[1] + 2)
        else:
            text_rect.topleft = (pos[0] + 2, pos[1] + 2)
        self.screen.blit(text_surface, text_rect)
        
        # Draw main text
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)
    
    @staticmethod
    def _lerp_color(c1, c2, t):
        t = np.clip(t, 0, 1)
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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


if __name__ == "__main__":
    # --- Manual Play Example ---
    # This block needs a display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Key mapping for manual control
    key_map = {
        pygame.K_UP: (1, 0, 0),
        pygame.K_DOWN: (2, 0, 0),
        pygame.K_LEFT: (3, 0, 0),
        pygame.K_RIGHT: (4, 0, 0),
        pygame.K_SPACE: (0, 1, 0),
        pygame.K_LSHIFT: (0, 0, 1),
        pygame.K_RSHIFT: (0, 0, 1),
    }

    # Pygame window for rendering
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fluid Flow Optimizer")
    clock = pygame.time.Clock()
    
    total_reward = 0.0
    
    print("\n--- Manual Control ---")
    print("Up/Down: Adjust Pipe 1 Diameter")
    print("Left/Right: Adjust Pipe 2 Diameter")
    print("Space/Shift: Adjust Pipe 3 Diameter")
    print("R: Reset Environment")
    print("Q: Quit")

    running = True
    while running:
        # Construct action from keyboard input
        action = [0, 0, 0] # [movement, space, shift]
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    print("--- Resetting Environment ---")
                    obs, info = env.reset()
                    total_reward = 0.0
                    done = False
        
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

    env.close()
    if info:
        print(f"\nGame finished. Final score: {info['score']}, Total reward: {total_reward:.2f}")