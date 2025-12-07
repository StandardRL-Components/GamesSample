import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:35:19.216568
# Source Brief: brief_00506.md
# Brief Index: 506
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent manipulates two types of liquids
    to fill target containers. The core mechanic involves using a "viscous"
    liquid that can be solidified to create barriers, controlling the flow
    of a "fast" liquid.

    The agent controls a cursor to direct the flow and solidification.
    - Up/Down: Select liquid type (viscous/fast).
    - Left/Right: Move the cursor.
    - Space: Pump the selected liquid.
    - Shift: Apply heat to solidify viscous liquid at the cursor.

    The goal is to fill five target areas with their designated liquid type
    within a time limit. The environment is designed with a focus on visual
    clarity and satisfying "game feel" through particle-based physics.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A physics-based puzzle game where you control the flow of two different liquids. "
        "Use a solidifying liquid to create barriers and guide a second liquid to fill target containers."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. ↑ also selects the viscous liquid, "
        "↓ selects the fast liquid. Press space to pump and hold shift to solidify."
    )
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_VISCOUS = (50, 150, 255)
    COLOR_FAST = (50, 255, 150)
    COLOR_SOLID = (20, 80, 130)
    COLOR_TARGET_EMPTY = (60, 60, 70)
    COLOR_TARGET_OUTLINE = (100, 100, 110)
    COLOR_TARGET_FILL_CORRECT = (200, 200, 220)
    COLOR_TARGET_FILL_WRONG = (220, 100, 100)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_HEAT_BAR = (255, 80, 80)
    COLOR_TEXT = (230, 230, 230)

    # Screen Dimensions
    WIDTH, HEIGHT = 640, 400

    # Game Parameters
    MAX_STEPS = 900  # Approx 30 seconds at 30 FPS
    CURSOR_SPEED = 6
    PUMP_RATE = 5  # Particles per pump action
    HEAT_RADIUS = 30
    HEAT_COST = 25
    HEAT_RECHARGE_RATE = 0.5
    MAX_HEAT = 100
    GRAVITY = pygame.Vector2(0, 0.2)
    MAX_PARTICLES = 1500

    class Particle:
        def __init__(self, pos, vel, p_type, color, radius):
            self.pos = pygame.Vector2(pos)
            self.vel = pygame.Vector2(vel)
            self.type = p_type
            self.color = color
            self.radius = radius

    class Target:
        def __init__(self, rect, required_type):
            self.rect = rect
            self.required_type = required_type
            self.max_volume = rect.width * rect.height / (math.pi * 3**2) # Approx particle area
            self.current_volumes = {'viscous': 0, 'fast': 0}
            self.is_complete = False
            self.was_complete = False
            self.last_correct_fill_percent = 0
            self.last_incorrect_fill_percent = 0

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_target = pygame.font.Font(None, 22)
        
        # Initialize state variables
        self.cursor_pos = pygame.Vector2(0, 0)
        self.selected_pump = 'viscous'
        self.heat = 0
        self.particles = []
        self.solids = []
        self.targets = []
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 4)
        self.selected_pump = 'viscous'
        self.heat = self.MAX_HEAT
        
        self.particles.clear()
        self.solids.clear()
        
        self._create_targets()
        
        return self._get_observation(), self._get_info()

    def _create_targets(self):
        self.targets.clear()
        target_zones = [
            pygame.Rect(50, 300, 100, 80),
            pygame.Rect(180, 320, 100, 60),
            pygame.Rect(360, 320, 100, 60),
            pygame.Rect(490, 300, 100, 80),
            pygame.Rect(265, 150, 110, 50) # A harder one
        ]
        
        types = ['viscous', 'fast']
        random.shuffle(types)
        required_types = [types[0], types[1], types[1], types[0], random.choice(types)]
        
        for i, zone in enumerate(target_zones):
            self.targets.append(self.Target(zone, required_types[i]))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_input(action)
        self._update_game_state()
        
        reward = self._calculate_reward()
        self.score += reward
        
        self.steps += 1
        terminated = self._check_termination()
        truncated = False # This environment does not truncate based on time limits
        
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Action 0: Movement
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        
        # Clamp cursor to screen
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

        # Use Up/Down (from movement) to select pump type
        if movement == 1: self.selected_pump = 'viscous'
        if movement == 2: self.selected_pump = 'fast'

        # Action 1: Pump
        if space_held:
            self._pump_liquid()

        # Action 2: Heat
        if shift_held:
            self._apply_heat()

    def _pump_liquid(self):
        # sound placeholder: # sfx_pump_loop()
        if len(self.particles) > self.MAX_PARTICLES:
            return

        p_type = self.selected_pump
        if p_type == 'viscous':
            color = self.COLOR_VISCOUS
            radius = 4
            for _ in range(self.PUMP_RATE):
                vel = pygame.Vector2(random.uniform(-0.8, 0.8), random.uniform(-1.5, -0.5))
                self.particles.append(self.Particle(self.cursor_pos, vel, p_type, color, radius))
        elif p_type == 'fast':
            color = self.COLOR_FAST
            radius = 3
            for _ in range(self.PUMP_RATE):
                vel = pygame.Vector2(random.uniform(-1.5, 1.5), random.uniform(-2.5, -1.0))
                self.particles.append(self.Particle(self.cursor_pos, vel, p_type, color, radius))

    def _apply_heat(self):
        if self.heat >= self.HEAT_COST:
            # sound placeholder: # sfx_heat_activate()
            self.heat -= self.HEAT_COST
            solidified_indices = []
            for i, p in enumerate(self.particles):
                if p.type == 'viscous' and self.cursor_pos.distance_to(p.pos) < self.HEAT_RADIUS:
                    solidified_indices.append(i)
                    self.solids.append(pygame.Rect(p.pos.x - p.radius, p.pos.y - p.radius, p.radius * 2, p.radius * 2))
            
            # Remove solidified particles by iterating backwards
            for i in sorted(solidified_indices, reverse=True):
                del self.particles[i]

    def _update_game_state(self):
        self._update_particles()
        self._update_targets()
        self.heat = min(self.MAX_HEAT, self.heat + self.HEAT_RECHARGE_RATE)
        
    def _update_particles(self):
        for p in self.particles[:]:
            p.vel += self.GRAVITY
            if p.type == 'viscous':
                p.vel *= 0.96 # High viscosity
            else: # fast
                p.vel *= 0.99 # Low viscosity
            
            p.pos += p.vel

            # Collision with solids for fast particles
            if p.type == 'fast':
                for solid_rect in self.solids:
                    if solid_rect.collidepoint(p.pos):
                        p.pos -= p.vel # Revert position
                        p.vel.y *= -0.5 # Bounce vertically
                        p.vel.x *= 0.8 # Lose horizontal speed
                        break

            # Boundary collisions
            if p.pos.x < p.radius or p.pos.x > self.WIDTH - p.radius:
                p.vel.x *= -0.7
                p.pos.x = np.clip(p.pos.x, p.radius, self.WIDTH - p.radius)
            if p.pos.y > self.HEIGHT - p.radius:
                self.particles.remove(p) # Remove particles that hit the floor
            elif p.pos.y < 0: # Remove particles that go off top
                self.particles.remove(p)


    def _update_targets(self):
        for t in self.targets:
            t.current_volumes = {'viscous': 0, 'fast': 0}

        for p in self.particles:
            for t in self.targets:
                if t.rect.collidepoint(p.pos):
                    t.current_volumes[p.type] += 1
                    break
        
        for t in self.targets:
            total_volume = sum(t.current_volumes.values())
            if total_volume >= t.max_volume:
                correct_volume = t.current_volumes.get(t.required_type, 0)
                if total_volume > 0 and correct_volume / total_volume > 0.85: # 85% purity to complete
                    t.is_complete = True
                else:
                    t.is_complete = False # Can become un-completed
            else:
                t.is_complete = False

    def _calculate_reward(self):
        reward = 0
        for t in self.targets:
            total_volume = sum(t.current_volumes.values())
            if total_volume == 0: continue

            correct_fill_percent = t.current_volumes.get(t.required_type, 0) / t.max_volume
            incorrect_fill_percent = (total_volume - t.current_volumes.get(t.required_type, 0)) / t.max_volume
            
            correct_fill_percent = min(correct_fill_percent, 1.0)
            incorrect_fill_percent = min(incorrect_fill_percent, 1.0)

            # Reward for increasing correct fill
            reward += (correct_fill_percent - t.last_correct_fill_percent) * 10.0 # +0.1 per 1%
            # Penalty for increasing incorrect fill
            reward -= (incorrect_fill_percent - t.last_incorrect_fill_percent) * 1.0 # -0.01 per 1%
            
            t.last_correct_fill_percent = correct_fill_percent
            t.last_incorrect_fill_percent = incorrect_fill_percent

            # Event-based reward for completing a target
            if t.is_complete and not t.was_complete:
                reward += 5
                # sound placeholder: # sfx_target_complete()
            t.was_complete = t.is_complete
        
        return reward

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            self.score -= 50 # Timeout penalty
            return True
        
        if all(t.is_complete for t in self.targets):
            self.score += 50 # Victory bonus
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render solids
        for s_rect in self.solids:
            pygame.draw.rect(self.screen, self.COLOR_SOLID, s_rect, border_radius=4)

        # Render particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos.x), int(p.pos.y), p.radius, p.color)
            pygame.gfxdraw.aacircle(self.screen, int(p.pos.x), int(p.pos.y), p.radius, p.color)

        # Render targets
        for t in self.targets:
            pygame.draw.rect(self.screen, self.COLOR_TARGET_EMPTY, t.rect, border_radius=8)
            
            total_fill = sum(t.current_volumes.values())
            if total_fill > 0:
                fill_ratio = min(total_fill / t.max_volume, 1.0)
                fill_height = int(t.rect.height * fill_ratio)
                fill_rect = pygame.Rect(t.rect.left, t.rect.bottom - fill_height, t.rect.width, fill_height)
                
                correct_ratio = t.current_volumes.get(t.required_type, 0) / total_fill if total_fill > 0 else 0
                fill_color = self.COLOR_TARGET_FILL_CORRECT if correct_ratio > 0.5 else self.COLOR_TARGET_FILL_WRONG
                pygame.draw.rect(self.screen, fill_color, fill_rect, border_bottom_left_radius=8, border_bottom_right_radius=8)

            pygame.draw.rect(self.screen, self.COLOR_TARGET_OUTLINE, t.rect, 2, border_radius=8)
            
            # Target info text
            icon_color = self.COLOR_VISCOUS if t.required_type == 'viscous' else self.COLOR_FAST
            pygame.draw.circle(self.screen, icon_color, (t.rect.centerx, t.rect.top + 15), 8)
            
            fill_percent = int(min(total_fill / t.max_volume, 1.0) * 100)
            text_surf = self.font_target.render(f"{fill_percent}%", True, self.COLOR_TEXT)
            self.screen.blit(text_surf, (t.rect.centerx - text_surf.get_width() // 2, t.rect.bottom - 25))

    def _render_ui(self):
        # Timer
        remaining_time = (self.MAX_STEPS - self.steps) / 30
        time_text = f"Time: {remaining_time:.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))

        # Score
        score_text = f"Score: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Heat bar
        heat_bar_bg = pygame.Rect(10, 40, 150, 20)
        pygame.draw.rect(self.screen, self.COLOR_TARGET_EMPTY, heat_bar_bg, border_radius=5)
        heat_fill_width = int(146 * (self.heat / self.MAX_HEAT))
        heat_fill_rect = pygame.Rect(12, 42, heat_fill_width, 16)
        pygame.draw.rect(self.screen, self.COLOR_HEAT_BAR, heat_fill_rect, border_radius=5)
        heat_text = self.font_target.render("HEAT", True, self.COLOR_TEXT)
        self.screen.blit(heat_text, (heat_bar_bg.centerx - heat_text.get_width() // 2, heat_bar_bg.centery - heat_text.get_height() // 2))

        # Selected pump indicator
        pump_text = f"Pump: {self.selected_pump.upper()}"
        pump_surf = self.font_ui.render(pump_text, True, self.COLOR_TEXT)
        pump_color = self.COLOR_VISCOUS if self.selected_pump == 'viscous' else self.COLOR_FAST
        pygame.draw.circle(self.screen, pump_color, (20, 90), 10)
        self.screen.blit(pump_surf, (40, 80))

        # Render cursor
        x, y = int(self.cursor_pos.x), int(self.cursor_pos.y)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x - 10, y), (x + 10, y), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x, y - 10), (x, y + 10), 2)
        
        # Check if shift is held for rendering heat radius
        # This is a bit of a hack since step() hasn't been called yet for the current frame
        # We can't reliably know the action. For a simple visual, this is okay.
        # A more robust way would be to pass action to render.
        # if shift_held := (self.action_space.sample()[2] == 1): # Example usage for rendering
        #     if self.heat >= self.HEAT_COST:
        #         pygame.gfxdraw.aacircle(self.screen, x, y, self.HEAT_RADIUS, self.COLOR_HEAT_BAR)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "completed_targets": sum(1 for t in self.targets if t.is_complete)
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Liquid Puzzle Environment")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    # Mapping keyboard keys to MultiDiscrete actions
    key_map = {
        pygame.K_UP:    [1, 0, 0],
        pygame.K_DOWN:  [2, 0, 0],
        pygame.K_LEFT:  [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
    }

    while not done:
        # Default action is no-op
        action = [0, 0, 0]
        
        keys = pygame.key.get_pressed()
        
        # Combine movement keys (only one direction at a time for simplicity)
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4

        # Space and Shift
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    env.close()