import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:31:33.277223
# Source Brief: brief_01788.md
# Brief Index: 1788
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class SedimentParticle:
    """Represents a single particle of sediment."""
    def __init__(self, pos, vel, color, density):
        self.pos = list(pos)
        self.vel = list(vel)
        self.color = color
        self.density = density
        self.life = random.uniform(150, 200) # Frames to live

    def update(self, gravity, current_vector):
        self.vel[1] += gravity * self.density
        self.vel[0] += current_vector[0]
        self.vel[1] += current_vector[1]
        
        # Damping
        self.vel[0] *= 0.98
        self.vel[1] *= 0.98

        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1

    def draw(self, surface):
        # Main particle
        pygame.draw.circle(surface, self.color, (int(self.pos[0]), int(self.pos[1])), 2)
        # Glow effect
        glow_color = (*self.color, 60)
        temp_surf = pygame.Surface((10, 10), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color, (5, 5), 5)
        surface.blit(temp_surf, (int(self.pos[0]) - 5, int(self.pos[1]) - 5), special_flags=pygame.BLEND_RGBA_ADD)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = "Strategically deposit different types of sediment to build stable structures against changing ocean currents."
    user_guide = "Controls: Use arrow keys (↑↓←→) to aim, press space to release sediment, and use shift to cycle sediment types."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_STEPS = 2000
        self.GRAVITY = 0.03
        
        # --- Colors ---
        self.COLOR_BG_TOP = (5, 10, 25)
        self.COLOR_BG_BOTTOM = (0, 0, 10)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_TEXT_SHADOW = (20, 20, 50)
        self.COLOR_BASE = (80, 80, 90)
        self.COLOR_AIMER = (255, 255, 0)
        self.COLOR_CURRENT_PARTICLE = (50, 150, 200, 30)

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
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.launcher_pos = (self.WIDTH // 2, 50)
        self.launcher_angle = -math.pi / 2
        self.plume_cooldown = 0
        self.last_shift_state = 0
        self.sediment_particles = []
        self.structure_blocks = {} # (gx, gy) -> color
        self.current_vector = [0.0, 0.0]
        self.target_current_vector = [0.0, 0.0]
        self.current_particles = []
        self.available_sediment_types = []
        self.selected_sediment_index = 0
        self.unlock_milestones = {}
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.launcher_angle = -math.pi / 2
        self.plume_cooldown = 0
        self.last_shift_state = 0

        self.sediment_particles = []
        self.structure_blocks = {}
        
        # Create base platform
        base_y = self.HEIGHT - self.GRID_SIZE * 3
        for i in range(10, self.WIDTH // self.GRID_SIZE - 10):
            self.structure_blocks[(i, base_y // self.GRID_SIZE)] = self.COLOR_BASE

        # Initialize currents
        self.current_vector = [0.0, 0.0]
        self.target_current_vector = [random.uniform(-0.05, 0.05), random.uniform(-0.02, 0.02)]
        self.current_particles = [
            [random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT)]
            for _ in range(150)
        ]

        # Initialize sediment types and unlocks
        self.available_sediment_types = [
            {'name': 'Sand', 'color': (210, 180, 140), 'density': 1.0},
            {'name': 'Clay', 'color': (139, 69, 19), 'density': 1.2},
        ]
        self.unlock_milestones = {
            100: {'type': 'sediment', 'data': {'name': 'Coral', 'color': (255, 100, 120), 'density': 0.8}, 'unlocked': False},
            500: {'type': 'sediment', 'data': {'name': 'Basalt', 'color': (70, 70, 80), 'density': 1.5}, 'unlocked': False},
            1000: {'type': 'sediment', 'data': {'name': 'Glow', 'color': (150, 255, 150), 'density': 0.6}, 'unlocked': False},
        }
        self.selected_sediment_index = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.game_over = self.game_over or self.steps >= self.MAX_STEPS

        if not self.game_over:
            self._handle_input(action)
            reward += self._update_game_state()
            self._check_progression()
            reward += self._check_unlocks()
        
        self.steps += 1
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Aiming
        aim_speed = 0.05
        if movement == 1: self.launcher_angle -= aim_speed # Up
        elif movement == 2: self.launcher_angle += aim_speed # Down
        elif movement == 3: self.launcher_angle -= aim_speed # Left
        elif movement == 4: self.launcher_angle += aim_speed # Right
        self.launcher_angle = max(-math.pi, min(0, self.launcher_angle))

        # Cycle sediment
        if shift_held and not self.last_shift_state:
            self.selected_sediment_index = (self.selected_sediment_index + 1) % len(self.available_sediment_types)
            # SFX: UI_click
        self.last_shift_state = shift_held

        # Release plume
        if space_held and self.plume_cooldown <= 0:
            self._spawn_plume()
            self.plume_cooldown = 15 # Cooldown of 15 frames
            # SFX: Plume_release_swoosh
        
        if self.plume_cooldown > 0:
            self.plume_cooldown -= 1

    def _spawn_plume(self):
        sediment_type = self.available_sediment_types[self.selected_sediment_index]
        for _ in range(20): # Plume of 20 particles
            angle_offset = random.uniform(-0.15, 0.15)
            speed = random.uniform(1.5, 2.5)
            vel = [
                speed * math.cos(self.launcher_angle + angle_offset),
                speed * math.sin(self.launcher_angle + angle_offset)
            ]
            self.sediment_particles.append(
                SedimentParticle(self.launcher_pos, vel, sediment_type['color'], sediment_type['density'])
            )

    def _update_game_state(self):
        step_reward = 0
        
        # Update current
        if self.steps % 200 == 0:
            max_speed = 0.05 + 0.05 * (self.steps // 500)
            self.target_current_vector = [random.uniform(-max_speed, max_speed), random.uniform(-max_speed/2, max_speed/2)]
        
        # Lerp current for smooth transition
        self.current_vector[0] += (self.target_current_vector[0] - self.current_vector[0]) * 0.01
        self.current_vector[1] += (self.target_current_vector[1] - self.current_vector[1]) * 0.01

        # Update sediment particles
        particles_to_remove = []
        for i, p in enumerate(self.sediment_particles):
            p.update(self.GRAVITY, self.current_vector)

            # Check for settling
            grid_pos = (int(p.pos[0] / self.GRID_SIZE), int(p.pos[1] / self.GRID_SIZE))
            if grid_pos[1] * self.GRID_SIZE >= self.HEIGHT - self.GRID_SIZE: # Hit bottom
                 particles_to_remove.append(i)
                 step_reward -= 0.01
                 continue
            
            support_pos = (grid_pos[0], grid_pos[1] + 1)
            if support_pos in self.structure_blocks and grid_pos not in self.structure_blocks:
                self.structure_blocks[grid_pos] = p.color
                self.score += 1
                step_reward += 0.1
                particles_to_remove.append(i)
                # SFX: Settle_sound
                continue

            # Check life and bounds
            if p.life <= 0 or not (0 < p.pos[0] < self.WIDTH):
                particles_to_remove.append(i)
                if p.pos[1] > self.launcher_pos[1]: # Only penalize if it fell past the launcher
                    step_reward -= 0.01

        # Remove particles in reverse to avoid index errors
        for i in sorted(particles_to_remove, reverse=True):
            del self.sediment_particles[i]

        # Check for structure collapse (simplified check)
        if self.steps % 30 == 0: # Check every 30 frames for performance
            unsupported = self._find_unsupported_blocks()
            if unsupported:
                # SFX: Collapse_rumble
                for block_pos in unsupported:
                    del self.structure_blocks[block_pos]
                
                # Simple collapse: if more than 5% of blocks fall, game over
                if len(unsupported) > len(self.structure_blocks) * 0.05 and len(self.structure_blocks) > 20:
                    step_reward -= 50
                    self.game_over = True

        return step_reward

    def _find_unsupported_blocks(self):
        unsupported = set()
        all_blocks = sorted(self.structure_blocks.keys(), key=lambda p: p[1])
        
        for pos in all_blocks:
            base_y = self.HEIGHT // self.GRID_SIZE - 3
            if pos[1] >= base_y:
                continue # Base blocks are always supported
            
            support_pos = (pos[0], pos[1] + 1)
            if support_pos not in self.structure_blocks:
                unsupported.add(pos)
        return unsupported

    def _check_progression(self):
        # This is where unlocks would happen based on score
        pass
    
    def _check_unlocks(self):
        reward = 0
        for milestone, data in self.unlock_milestones.items():
            if self.score >= milestone and not data['unlocked']:
                if data['type'] == 'sediment':
                    self.available_sediment_types.append(data['data'])
                    reward += 10 # Goal-oriented reward
                self.unlock_milestones[milestone]['unlocked'] = True
                # SFX: Unlock_chime
        return reward

    def _get_observation(self):
        # Draw background gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render current particles
        for p in self.current_particles:
            p[0] += self.current_vector[0] * 50
            p[1] += self.current_vector[1] * 50
            p[0] %= self.WIDTH
            if p[1] > self.HEIGHT: p[1] = 0
            if p[1] < 0: p[1] = self.HEIGHT
            
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            temp_surf.fill(self.COLOR_CURRENT_PARTICLE)
            self.screen.blit(temp_surf, (int(p[0]), int(p[1])), special_flags=pygame.BLEND_RGBA_ADD)

        # Render structure
        for pos, color in self.structure_blocks.items():
            rect = pygame.Rect(pos[0] * self.GRID_SIZE, pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            # Use slightly darker color for border
            border_color = tuple(max(0, c-20) for c in color)
            pygame.draw.rect(self.screen, border_color, rect)
            inner_rect = rect.inflate(-2, -2)
            pygame.draw.rect(self.screen, color, inner_rect)

        # Render sediment particles
        for p in self.sediment_particles:
            p.draw(self.screen)

        # Render aimer
        aimer_len = 30
        end_pos = (
            self.launcher_pos[0] + aimer_len * math.cos(self.launcher_angle),
            self.launcher_pos[1] + aimer_len * math.sin(self.launcher_angle)
        )
        pygame.draw.line(self.screen, self.COLOR_AIMER, self.launcher_pos, end_pos, 2)
        pygame.gfxdraw.filled_circle(self.screen, int(self.launcher_pos[0]), int(self.launcher_pos[1]), 5, self.COLOR_AIMER)
        pygame.gfxdraw.aacircle(self.screen, int(self.launcher_pos[0]), int(self.launcher_pos[1]), 5, self.COLOR_AIMER)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, pos, font, color, shadow_color):
            shadow = font.render(text, True, shadow_color)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            main_text = font.render(text, True, color)
            self.screen.blit(main_text, pos)

        # Score and Steps
        draw_text(f"SCORE: {self.score}", (15, 10), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        draw_text(f"STEPS: {self.steps}/{self.MAX_STEPS}", (15, 35), self.font_small, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Sediment selector
        draw_text("SEDIMENT:", (self.WIDTH - 200, 10), self.font_small, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        for i, s_type in enumerate(self.available_sediment_types):
            rect = pygame.Rect(self.WIDTH - 200 + i * 30, 30, 25, 25)
            pygame.draw.rect(self.screen, s_type['color'], rect, border_radius=4)
            if i == self.selected_sediment_index:
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=4)

        # Current indicator
        center = (self.WIDTH - 50, self.HEIGHT - 50)
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], 20, self.COLOR_TEXT)
        current_mag = math.hypot(self.current_vector[0], self.current_vector[1])
        if current_mag > 1e-6:
            end_pos = (
                center[0] + self.current_vector[0] / current_mag * 18,
                center[1] + self.current_vector[1] / current_mag * 18
            )
            pygame.draw.line(self.screen, self.COLOR_TEXT, center, end_pos, 2)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sediment_count": len(self.available_sediment_types),
            "structure_size": len(self.structure_blocks),
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == "__main__":
    # The original code had an issue here: it called validate_implementation
    # in __init__ before reset was called, which would fail.
    # The logic is moved here for clarity.
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    try:
        env.validate_implementation()
    except AssertionError as e:
        print(f"Validation failed: {e}")

    # --- Manual Play Setup ---
    # Un-comment the following lines to run with a display
    # os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    pygame.display.set_caption("Sediment Builder")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Key mapping for human play
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    print("\n--- Controls ---")
    print("Arrow Keys: Aim")
    print("Spacebar: Release Plume")
    print("Shift: Cycle Sediment Type")
    print("R: Reset Environment")
    print("----------------\n")

    while not done:
        # Action defaults
        movement = 0
        space_held = 0
        shift_held = 0

        # Pygame event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                print("Environment Reset.")

        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize first key in map
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}")
            obs, info = env.reset()

        # Render to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()