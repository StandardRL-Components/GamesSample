import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:01:50.103661
# Source Brief: brief_01429.md
# Brief Index: 1429
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: A real-time strategy resource management game.
    The player must gather three types of resources to build and upgrade
    five structures. The goal is to build all five structures before the
    time limit of 12000 steps (2 minutes) runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A real-time strategy game where you manage resources to build and upgrade structures against a time limit."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to select a structure. Press space to build or upgrade the selected structure."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 100  # Affects game speed and timer calculations

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GREEN = (50, 205, 50)
        self.COLOR_BLUE = (65, 105, 225)
        self.COLOR_RED = (220, 20, 60)
        self.RESOURCE_COLORS = [self.COLOR_GREEN, self.COLOR_BLUE, self.COLOR_RED]
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_STRUCTURE_BG = (40, 45, 65)
        self.COLOR_STRUCTURE_OUTLINE = (70, 80, 110)
        self.COLOR_STRUCTURE_BUILT = (120, 130, 160)
        self.COLOR_STRUCTURE_UPGRADED = (200, 210, 240)
        self.COLOR_SELECT = (255, 215, 0)
        self.COLOR_TIMER_DANGER = (255, 60, 60)
        self.COLOR_CANNOT_AFFORD = (255, 80, 80)

        # Game Parameters
        self.MAX_STEPS = 12000
        self.NUM_STRUCTURES = 5
        self.WIN_CONDITION_COUNT = 5
        self.BASE_RESOURCE_REGEN = np.array([0.1, 0.05, 0.02])
        self.STRUCTURE_COSTS = np.array([
            [10, 5, 2],   # Structure 1
            [12, 4, 3],   # Structure 2
            [8, 6, 1],    # Structure 3
            [15, 3, 4],   # Structure 4
            [11, 7, 2]    # Structure 5
        ])
        self.STRUCTURE_UPGRADE_MULTIPLIER = 2.0
        self.STRUCTURE_GEN_BONUS = 0.05
        self.STRUCTURE_RESOURCE_MAP = [0, 1, 2, 0, 1]  # Maps structure index to resource index it generates

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
        self.ui_font_small = pygame.font.SysFont("sans-serif", 16)
        self.ui_font_medium = pygame.font.SysFont("sans-serif", 24)
        self.ui_font_large = pygame.font.SysFont("sans-serif", 32)
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.resources = np.zeros(3, dtype=float)
        self.structures = [0] * self.NUM_STRUCTURES
        self.selected_structure_idx = 0
        self._previous_space_held = False
        self._previous_movement = 0
        self.particles = []
        self.action_feedback = {} # For temporary visual effects

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.resources = np.array([5.0, 5.0, 5.0]) # Start with some resources
        self.structures = [0] * self.NUM_STRUCTURES # 0: unbuilt, 1: built, 2: upgraded
        self.selected_structure_idx = 0
        self._previous_space_held = False
        self._previous_movement = 0
        self.particles = []
        self.action_feedback = {}

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = self._update_game_state(action)
        self.score += reward
        self.steps += 1
        
        terminated, terminal_reward = self._check_termination()
        self.game_over = terminated
        reward += terminal_reward
        self.score += terminal_reward

        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _update_game_state(self, action):
        """Handle all game logic for one step."""
        reward = 0.0
        
        # 1. Update resources
        base_gen = self.BASE_RESOURCE_REGEN
        bonus_gen = np.zeros(3, dtype=float)
        for i, level in enumerate(self.structures):
            if level > 0:
                resource_idx = self.STRUCTURE_RESOURCE_MAP[i]
                # Level 1 gives 1x bonus, Level 2 gives 2x bonus
                bonus_gen[resource_idx] += self.STRUCTURE_GEN_BONUS * level
        
        total_gen = base_gen + bonus_gen
        self.resources += total_gen
        reward += 0.01 * np.sum(total_gen) # Continuous reward for generation

        # 2. Handle player input
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Handle selection (on press, not hold)
        movement_pressed = movement != 0 and movement != self._previous_movement
        if movement_pressed:
            if movement == 4: # Right
                self.selected_structure_idx = (self.selected_structure_idx + 1) % self.NUM_STRUCTURES
            elif movement == 3: # Left
                self.selected_structure_idx = (self.selected_structure_idx - 1 + self.NUM_STRUCTURES) % self.NUM_STRUCTURES

        # Handle build/upgrade action (on press, not hold)
        space_pressed = space_held and not self._previous_space_held
        if space_pressed:
            build_reward = self._attempt_build_or_upgrade()
            reward += build_reward

        self._previous_space_held = space_held
        self._previous_movement = movement

        # 3. Update particles
        self._update_particles()
        
        # 4. Update feedback timers
        keys_to_delete = []
        for key, data in self.action_feedback.items():
            data['timer'] -= 1
            if data['timer'] <= 0:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del self.action_feedback[key]

        return reward

    def _attempt_build_or_upgrade(self):
        idx = self.selected_structure_idx
        level = self.structures[idx]
        pos = self._get_structure_pos(idx)

        if level == 0: # Attempt to build
            cost = self.STRUCTURE_COSTS[idx]
            if np.all(self.resources >= cost):
                self.resources -= cost
                self.structures[idx] = 1
                self._spawn_particles(pos, self.COLOR_STRUCTURE_BUILT, 30)
                return 1.0 # Build reward
            else:
                self.action_feedback[idx] = {'timer': 20, 'color': self.COLOR_CANNOT_AFFORD}
                return 0.0
        
        elif level == 1: # Attempt to upgrade
            cost = self.STRUCTURE_COSTS[idx] * self.STRUCTURE_UPGRADE_MULTIPLIER
            if np.all(self.resources >= cost):
                self.resources -= cost
                self.structures[idx] = 2
                self._spawn_particles(pos, self.COLOR_SELECT, 50, 2.5)
                return 2.0 # Upgrade reward
            else:
                self.action_feedback[idx] = {'timer': 20, 'color': self.COLOR_CANNOT_AFFORD}
                return 0.0
        
        return 0.0

    def _check_termination(self):
        """Check for win/loss conditions."""
        # Win condition
        built_count = sum(1 for s in self.structures if s > 0)
        if built_count >= self.WIN_CONDITION_COUNT:
            return True, 100.0
        
        # Loss condition (timeout is handled by truncated flag)
        if self.steps >= self.MAX_STEPS:
            return True, -100.0
            
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Render the main game elements like structures and particles."""
        self._render_particles()
        
        for i in range(self.NUM_STRUCTURES):
            self._render_structure_slot(i)

    def _render_ui(self):
        """Render the HUD elements."""
        self._render_resource_bars()
        self._render_timer()
        self._render_structure_count()
        self._render_selected_info()

    def _render_structure_slot(self, i):
        x, y = self._get_structure_pos(i)
        w, h = 80, 80
        rect = pygame.Rect(x - w/2, y - h/2, w, h)
        
        # Background
        pygame.draw.rect(self.screen, self.COLOR_STRUCTURE_BG, rect, border_radius=8)

        # Fill based on level
        level = self.structures[i]
        if level == 1:
            pygame.draw.rect(self.screen, self.COLOR_STRUCTURE_BUILT, rect.inflate(-10, -10), border_radius=6)
        elif level == 2:
            pygame.draw.rect(self.screen, self.COLOR_STRUCTURE_UPGRADED, rect.inflate(-10, -10), border_radius=6)
            # Add a visual flair for upgraded
            pygame.gfxdraw.filled_polygon(self.screen, [(x, y-15), (x-13, y+8), (x+13, y+8)], self.COLOR_SELECT)


        # Outline and selection highlight
        is_selected = (i == self.selected_structure_idx)
        outline_color = self.COLOR_STRUCTURE_OUTLINE
        outline_width = 2
        
        if is_selected:
            # Pulsing glow effect for selection
            pulse = abs(math.sin(self.steps * 0.1))
            outline_color = self._lerp_color(self.COLOR_SELECT, self.COLOR_TEXT, pulse)
            outline_width = 3

        pygame.draw.rect(self.screen, outline_color, rect, width=outline_width, border_radius=8)
        
        # "Can't afford" flash
        if i in self.action_feedback:
            feedback_alpha = self.action_feedback[i]['timer'] / 20.0
            self._draw_rect_alpha(self.screen, (*self.action_feedback[i]['color'], int(180 * feedback_alpha)), rect, 8)

        # Label
        label_text = self.ui_font_small.render(f"S{i+1}", True, self.COLOR_TEXT)
        self.screen.blit(label_text, (rect.centerx - label_text.get_width()/2, rect.top + 5))

    def _render_resource_bars(self):
        bar_w, bar_h = 180, 20
        start_x, start_y = 20, 50
        max_res_display = 50.0 # Visual cap for the bar
        
        for i in range(3):
            y = start_y + i * 40
            
            # Label
            label_surf = self.ui_font_medium.render(f"R{i+1}", True, self.RESOURCE_COLORS[i])
            self.screen.blit(label_surf, (start_x, y - 28))
            
            # Bar background
            bg_rect = pygame.Rect(start_x, y, bar_w, bar_h)
            pygame.draw.rect(self.screen, self.COLOR_STRUCTURE_BG, bg_rect, border_radius=5)
            
            # Bar fill
            fill_ratio = min(1.0, self.resources[i] / max_res_display)
            fill_rect = pygame.Rect(start_x, y, int(bar_w * fill_ratio), bar_h)
            pygame.draw.rect(self.screen, self.RESOURCE_COLORS[i], fill_rect, border_radius=5)
            
            # Outline
            pygame.draw.rect(self.screen, self.COLOR_STRUCTURE_OUTLINE, bg_rect, width=2, border_radius=5)

            # Numeric value
            val_text = f"{self.resources[i]:.1f}"
            val_surf = self.ui_font_small.render(val_text, True, self.COLOR_TEXT)
            self.screen.blit(val_surf, (start_x + bar_w + 10, y + 2))
            
    def _render_timer(self):
        remaining_steps = self.MAX_STEPS - self.steps
        remaining_seconds = max(0, remaining_steps / self.FPS)
        minutes = int(remaining_seconds // 60)
        seconds = int(remaining_seconds % 60)
        
        time_str = f"{minutes:02}:{seconds:02}"
        color = self.COLOR_TEXT
        
        # Flash red when time is low
        if remaining_seconds < 10 and self.steps % 50 < 25:
             color = self.COLOR_TIMER_DANGER
             
        time_surf = self.ui_font_large.render(time_str, True, color)
        self.screen.blit(time_surf, (self.WIDTH/2 - time_surf.get_width()/2, 10))

    def _render_structure_count(self):
        built_count = sum(1 for s in self.structures if s > 0)
        count_str = f"Built: {built_count} / {self.WIN_CONDITION_COUNT}"
        count_surf = self.ui_font_medium.render(count_str, True, self.COLOR_TEXT)
        self.screen.blit(count_surf, (self.WIDTH - count_surf.get_width() - 20, self.HEIGHT - count_surf.get_height() - 10))

    def _render_selected_info(self):
        idx = self.selected_structure_idx
        level = self.structures[idx]
        x, y = self._get_structure_pos(idx)
        
        info_y = y + 55
        
        if level == 0:
            text = "Cost:"
            cost = self.STRUCTURE_COSTS[idx]
        elif level == 1:
            text = "Upgrade Cost:"
            cost = self.STRUCTURE_COSTS[idx] * self.STRUCTURE_UPGRADE_MULTIPLIER
        else: # level 2
            text = "Max Level"
            cost = None

        title_surf = self.ui_font_medium.render(text, True, self.COLOR_SELECT)
        self.screen.blit(title_surf, (x - title_surf.get_width()/2, info_y))
        
        if cost is not None:
            for i, res_cost in enumerate(cost):
                has_enough = self.resources[i] >= res_cost
                color = self.RESOURCE_COLORS[i] if has_enough else self.COLOR_CANNOT_AFFORD
                cost_str = f"{res_cost:.0f}"
                cost_surf = self.ui_font_small.render(cost_str, True, color)
                
                # Position costs horizontally
                offset_x = (i - 1) * 40
                self.screen.blit(cost_surf, (x - cost_surf.get_width()/2 + offset_x, info_y + 30))

    def _get_structure_pos(self, i):
        num_slots = self.NUM_STRUCTURES
        padding = 40
        slot_width = (self.WIDTH - 2 * padding) / num_slots
        x = padding + slot_width * (i + 0.5)
        y = self.HEIGHT / 2
        return int(x), int(y)

    def _spawn_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(20, 40)
            radius = random.uniform(2, 5)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'max_life': lifespan, 'color': color, 'radius': radius})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_life']))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'] * (p['lifespan'] / p['max_life']))
            if radius > 0:
                self._draw_circle_alpha(self.screen, color, pos, radius)
                
    def _lerp_color(self, c1, c2, t):
        return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

    def _draw_rect_alpha(self, surface, color, rect, border_radius):
        shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, color, shape_surf.get_rect(), border_radius=border_radius)
        surface.blit(shape_surf, rect)

    def _draw_circle_alpha(self, surface, color, center, radius):
        target_rect = pygame.Rect(center[0] - radius, center[1] - radius, 2 * radius, 2 * radius)
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(shape_surf, color, (radius, radius), radius)
        surface.blit(shape_surf, target_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "resources": self.resources,
            "structures": self.structures,
            "built_count": sum(1 for s in self.structures if s > 0)
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, ensure you have pygame installed and remove the SDL_VIDEODRIVER dummy line
    # or run in an environment where a display is available.
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.font.init() # Re-init font system after unsetting dummy driver
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a display
    display_surface = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("GameEnv Test")

    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # The test script uses held keys, so we'll do the same for consistency
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_RIGHT:
            #         action[0] = 4
            #     elif event.key == pygame.K_LEFT:
            #         action[0] = 3
            #     elif event.key == pygame.K_SPACE:
            #         action[1] = 1

        # Use get_pressed for continuous actions
        keys = pygame.key.get_pressed()
        
        # Map keyboard to MultiDiscrete action space
        # Movement
        if keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        # The original code had up/down mapping, but the step function doesn't use them for selection.
        # We will keep the mapping for potential debugging but note they have no effect on selection.
        elif keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        else:
            action[0] = 0
            
        # Space button
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        # Shift button (unused in core logic)
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the screen
        game_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_surface.blit(game_surface, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            
        env.clock.tick(60) # Control human play speed

    env.close()