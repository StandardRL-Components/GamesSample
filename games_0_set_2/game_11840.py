import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:36:50.464314
# Source Brief: brief_01840.md
# Brief Index: 1840
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Petal:
    """Represents a single launched petal card."""
    def __init__(self, pos, vel, p_type_idx, color):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.p_type_idx = p_type_idx
        self.color = color
        self.is_settled = False
        self.settle_timer = 0
        self.radius = 12

    def update(self, other_petals, magnetism_level, bounds):
        if self.is_settled:
            self.vel *= 0 # Ensure it stays put
            return

        # Gravity
        self.vel[1] += 0.05

        # Magnetic forces
        for other in other_petals:
            if other is self:
                continue
            dist_vec = other.pos - self.pos
            dist_sq = np.dot(dist_vec, dist_vec)
            if dist_sq < 1:
                dist_sq = 1 # Avoid division by zero
            
            # Repulsion at very close range to prevent overlap
            if dist_sq < (self.radius * 2)**2:
                 force_mag = -10 / dist_sq
            else:
                # Attraction based on magnetism level
                force_mag = (magnetism_level * 100) / dist_sq
            
            force_vec = dist_vec / np.sqrt(dist_sq) * force_mag
            self.vel += force_vec

        # Damping
        self.vel *= 0.98
        self.pos += self.vel

        # Boundary collision
        if self.pos[0] < self.radius:
            self.pos[0] = self.radius
            self.vel[0] *= -0.5
        if self.pos[0] > bounds[0] - self.radius:
            self.pos[0] = bounds[0] - self.radius
            self.vel[0] *= -0.5
        if self.pos[1] < self.radius:
            self.pos[1] = self.radius
            self.vel[1] *= -0.5
        if self.pos[1] > bounds[1] - self.radius:
            self.pos[1] = bounds[1] - self.radius
            self.vel[1] *= -0.5
            # Settle on the floor
            if np.linalg.norm(self.vel) < 1.0:
                 self.vel[1] = 0

        # Check for settling
        if np.linalg.norm(self.vel) < 0.1:
            self.settle_timer += 1
            if self.settle_timer > 30: # Must be still for 1 second
                self.is_settled = True
                self.vel = np.array([0.0, 0.0])
                # sfx_settle
        else:
            self.settle_timer = 0

    def draw(self, surface):
        x, y = int(self.pos[0]), int(self.pos[1])
        
        # Glow effect
        glow_color = self.color + (50,) # Add alpha
        glow_radius = int(self.radius * 1.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Main petal shape
        points = []
        for i in range(5):
            angle = math.pi * 2 * i / 5 - math.pi / 2
            points.append((x + math.cos(angle) * self.radius, y + math.sin(angle) * self.radius))
        
        pygame.gfxdraw.aapolygon(surface, points, self.color)
        pygame.gfxdraw.filled_polygon(surface, points, self.color)
        
        # Highlight
        pygame.gfxdraw.pixel(surface, x, y - self.radius // 2, (255, 255, 255, 150))


class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, pos, vel, color, lifespan):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.color = color
        self.lifespan = lifespan
        self.life = lifespan

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.lifespan))
            color_with_alpha = self.color + (alpha,)
            pygame.draw.circle(surface, color_with_alpha, self.pos.astype(int), int(self.life / self.lifespan * 3 + 1))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Launch colorful petals into a magnetic field to create a beautiful garden. "
        "Carefully aim and adjust the magnetic force to guide the petals into their target locations."
    )
    user_guide = (
        "Controls: Use ↑/↓ to aim left/right, and ←/→ to adjust magnetism. "
        "Press space to launch and shift to switch petals."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLAY_AREA_HEIGHT = 320
    MAX_STEPS = 1500
    
    COLOR_BG_TOP = (10, 5, 25)
    COLOR_BG_BOTTOM = (20, 10, 40)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_TARGET_GHOST = (255, 255, 255, 30)

    PETAL_TYPES = [
        {"name": "Rose", "color": (255, 50, 100)},
        {"name": "Viola", "color": (150, 80, 255)},
        {"name": "Marigold", "color": (255, 180, 40)},
    ]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # Game state variables are initialized in reset()
        self.launcher_pos = np.array([self.SCREEN_WIDTH / 2, self.PLAY_AREA_HEIGHT - 20], dtype=float)
        self.last_space_held = False
        self.last_shift_held = False

        self.reset()
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.launcher_angle = -math.pi / 2
        self.magnetism_level = 0.1
        
        self.placed_petals = []
        self.particles = []
        
        self.current_petal_idx = 0
        self.petal_inventory = [5, 5, 5] # 5 of each type
        
        self._create_target_pattern()
        
        self.last_space_held = True # Prevent launch on first frame
        self.last_shift_held = True # Prevent switch on first frame

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Action Processing ---
        # Adjust angle
        if movement == 1: self.launcher_angle -= 0.05
        if movement == 2: self.launcher_angle += 0.05
        self.launcher_angle = np.clip(self.launcher_angle, -math.pi, 0)
        
        # Adjust magnetism
        if movement == 3: self.magnetism_level -= 0.02
        if movement == 4: self.magnetism_level += 0.02
        self.magnetism_level = np.clip(self.magnetism_level, 0.0, 1.0)

        # Cycle petal type
        if shift_held and not self.last_shift_held:
            self.current_petal_idx = (self.current_petal_idx + 1) % len(self.PETAL_TYPES)
            # sfx_switch
        
        # Launch petal
        if space_held and not self.last_space_held and self.petal_inventory[self.current_petal_idx] > 0:
            self._launch_petal()
            # sfx_launch

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        # --- Game Logic Update ---
        self._update_physics()
        self._update_particles()
        
        reward = self._check_settled_petals_and_reward()
        terminated, term_reward = self._check_termination()
        reward += term_reward
        
        self.steps += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _launch_petal(self):
        self.petal_inventory[self.current_petal_idx] -= 1
        launch_power = 8.0
        vel = np.array([math.cos(self.launcher_angle), math.sin(self.launcher_angle)]) * launch_power
        petal_info = self.PETAL_TYPES[self.current_petal_idx]
        
        new_petal = Petal(
            pos=self.launcher_pos.copy(),
            vel=vel,
            p_type_idx=self.current_petal_idx,
            color=petal_info["color"]
        )
        self.placed_petals.append(new_petal)
        
        for _ in range(15):
            p_vel = vel * 0.2 + self.np_random.standard_normal(2) * 0.5
            self.particles.append(Particle(self.launcher_pos.copy(), p_vel, petal_info["color"], 30))

    def _update_physics(self):
        for petal in self.placed_petals:
            petal.update(self.placed_petals, self.magnetism_level, (self.SCREEN_WIDTH, self.PLAY_AREA_HEIGHT))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _check_settled_petals_and_reward(self):
        reward = 0
        for petal in self.placed_petals:
            if petal.is_settled and not hasattr(petal, 'reward_given'):
                petal.reward_given = True
                
                best_dist = float('inf')
                best_target_idx = -1

                for i, target in enumerate(self.target_pattern):
                    if not target['filled'] and target['type_idx'] == petal.p_type_idx:
                        dist = np.linalg.norm(petal.pos - target['pos'])
                        if dist < best_dist:
                            best_dist = dist
                            best_target_idx = i
                
                if best_target_idx != -1:
                    if best_dist < 20: # Success threshold
                        self.target_pattern[best_target_idx]['filled'] = True
                        self.target_pattern[best_target_idx]['filled_pos'] = petal.pos
                        self.score += 10
                        reward += 1.0
                        # sfx_success_ding
                        for _ in range(30):
                            angle = self.np_random.uniform(0, 2 * math.pi)
                            p_vel = np.array([math.cos(angle), math.sin(angle)]) * self.np_random.uniform(1, 3)
                            self.particles.append(Particle(petal.pos.copy(), p_vel, petal.color, 40))
                    else:
                        reward -= 0.1 # Penalty for missing
                        # sfx_fail_thud
                else:
                    reward -= 0.1 # Penalty for wrong type or no available slots
        return reward

    def _check_termination(self):
        # Win condition: all targets filled
        if all(t['filled'] for t in self.target_pattern):
            self.game_over = True
            return True, 50.0

        # Loss condition: out of cards and all are settled
        cards_left = sum(self.petal_inventory)
        all_settled = all(p.is_settled for p in self.placed_petals)
        if cards_left == 0 and all_settled:
            self.game_over = True
            return True, -50.0

        # Max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True, 0.0
            
        return False, 0.0

    def _create_target_pattern(self):
        self.target_pattern = []
        num_targets = 3 + self.score // 25
        num_targets = min(num_targets, 8)

        center_x, center_y = self.SCREEN_WIDTH / 2, 120
        for i in range(num_targets):
            angle = 2 * math.pi * i / num_targets
            radius = 60 + (i % 2) * 30
            pos = np.array([
                center_x + math.cos(angle) * radius,
                center_y + math.sin(angle) * radius
            ])
            type_idx = self.np_random.integers(0, len(self.PETAL_TYPES))
            self.target_pattern.append({'pos': pos, 'type_idx': type_idx, 'filled': False})

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Rendering Methods ---
    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            mix = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - mix) + self.COLOR_BG_BOTTOM[0] * mix),
                int(self.COLOR_BG_TOP[1] * (1 - mix) + self.COLOR_BG_BOTTOM[1] * mix),
                int(self.COLOR_BG_TOP[2] * (1 - mix) + self.COLOR_BG_BOTTOM[2] * mix),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        self._render_magnetic_field()
        self._render_target_ghosts()
        for p in self.particles:
            p.draw(self.screen)
        for petal in self.placed_petals:
            petal.draw(self.screen)

    def _render_magnetic_field(self):
        # Shimmering lines visualizing the field potential
        for i in range(10):
            points = []
            for x in range(0, self.SCREEN_WIDTH, 10):
                offset = math.sin(x / 50 + self.steps / 30.0 + i * 0.5) * 10
                y = i * (self.PLAY_AREA_HEIGHT / 10) + offset
                points.append((x, y))
            if len(points) > 1:
                alpha = int(self.magnetism_level * 30 + 5)
                pygame.draw.aalines(self.screen, (100, 150, 255, alpha), False, points)

    def _render_target_ghosts(self):
        for target in self.target_pattern:
            if not target['filled']:
                pos = target['pos'].astype(int)
                color = self.PETAL_TYPES[target['type_idx']]['color']
                ghost_color = (color[0], color[1], color[2], 30)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, ghost_color)
            else:
                # Draw a connection line from target to filled position
                start_pos = target['pos'].astype(int)
                end_pos = target['filled_pos'].astype(int)
                pygame.draw.aaline(self.screen, (255,255,255,50), start_pos, end_pos)

    def _render_ui(self):
        # Draw UI background panel
        ui_rect = pygame.Rect(0, self.PLAY_AREA_HEIGHT, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.PLAY_AREA_HEIGHT)
        pygame.draw.rect(self.screen, (0,0,0,100), ui_rect)
        pygame.draw.line(self.screen, (100, 150, 255, 100), (0, self.PLAY_AREA_HEIGHT), (self.SCREEN_WIDTH, self.PLAY_AREA_HEIGHT))

        self._render_launcher()
        self._render_hud()

    def _render_launcher(self):
        # Launcher base
        pygame.draw.circle(self.screen, (80, 100, 150), self.launcher_pos.astype(int), 10)
        
        # Launcher barrel
        end_pos = self.launcher_pos + np.array([math.cos(self.launcher_angle), math.sin(self.launcher_angle)]) * 30
        pygame.draw.aaline(self.screen, (150, 180, 255), self.launcher_pos, end_pos, 2)

    def _render_hud(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Petal Selection
        petal_info = self.PETAL_TYPES[self.current_petal_idx]
        petal_count = self.petal_inventory[self.current_petal_idx]
        
        select_text = self.font_main.render(f"Petal: {petal_info['name']} (x{petal_count})", True, self.COLOR_UI_TEXT)
        self.screen.blit(select_text, (10, self.PLAY_AREA_HEIGHT + 15))

        # Magnetism Meter
        mag_text = self.font_small.render("Magnetism", True, self.COLOR_UI_TEXT)
        self.screen.blit(mag_text, (10, self.PLAY_AREA_HEIGHT + 50))
        
        bar_x, bar_y, bar_w, bar_h = 10, self.PLAY_AREA_HEIGHT + 65, 200, 15
        pygame.draw.rect(self.screen, (30, 30, 60), (bar_x, bar_y, bar_w, bar_h))
        fill_w = bar_w * self.magnetism_level
        fill_color = (100 + 155 * self.magnetism_level, 100, 255 - 155 * self.magnetism_level)
        pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, fill_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_w, bar_h), 1)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and is not part of the gym environment
    # It requires a display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    pygame.display.init()
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Magnetic Petal Garden")
    game_clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # --- Human Input to Action Mapping ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        # We need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        game_clock.tick(60) # Limit frame rate
        
    env.close()