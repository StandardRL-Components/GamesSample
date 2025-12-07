import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:35:00.140314
# Source Brief: brief_03061.md
# Brief Index: 3061
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, life=30):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.max_life = life
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 3)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        return self.life > 0

    def draw(self, surface):
        alpha = int(255 * (self.life / self.max_life))
        if alpha > 0:
            radius = int(3 * (self.life / self.max_life))
            color = (*self.color, alpha)
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            surface.blit(temp_surf, (int(self.x - radius), int(self.y - radius)), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Manage two robotic arms on a fast-paced assembly line. "
        "Adjust their speed and precision to assemble widgets against the clock."
    )
    user_guide = (
        "Controls: Use ↑↓ to adjust arm speed and ←→ for precision. "
        "Press 'shift' to switch between arms and 'space' to start assembling a widget."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.render_mode = render_mode
        pygame.init()
        pygame.font.init()
        self.width, self.height = 640, 400
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.fps = 30

        # --- Visuals & Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_BELT = (30, 40, 55)
        self.COLOR_TEXT = (220, 230, 240)
        self.COLOR_ACTIVE = (50, 150, 255)
        self.COLOR_INACTIVE = (80, 90, 110)
        self.COLOR_SUCCESS = (70, 220, 120)
        self.COLOR_FAILURE = (255, 80, 80)
        self.COLOR_PROGRESS = (255, 180, 50)
        
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_icon = pygame.font.SysFont("Consolas", 30, bold=True)

        # --- Game Constants ---
        self.MAX_TIME = 90.0  # seconds
        self.WIDGETS_TO_WIN = 10
        self.MAX_STEPS = int(self.MAX_TIME * self.fps)
        self.NUM_WIDGETS = 10
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0.0
        self.completed_widgets = 0
        self.active_arm_idx = 0
        self.arms = []
        self.widgets = []
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False

        self._initialize_layout()
        
        # --- Critical Self-Check ---
        # self.validate_implementation() # Commented out for submission

    def _initialize_layout(self):
        """Pre-calculate positions for UI elements."""
        self.arm_positions = [
            pygame.Vector2(80, self.height // 2),
            pygame.Vector2(self.width - 80, self.height // 2)
        ]
        
        belt_y = self.height // 2 - 30
        belt_height = 60
        self.belt_rect = pygame.Rect(0, belt_y, self.width, belt_height)
        
        widget_spacing = (self.width - 100) / (self.NUM_WIDGETS - 1)
        self.widget_positions = [
            pygame.Vector2(50 + i * widget_spacing, self.height // 2)
            for i in range(self.NUM_WIDGETS)
        ]
        self.widget_size = pygame.Vector2(40, 40)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_TIME
        self.completed_widgets = 0
        self.active_arm_idx = 0
        
        self.arms = [
            {'speed': 3, 'precision': 80, 'busy_until': 0, 'target_idx': None}
            for _ in range(2)
        ]
        
        self.widgets = [
            {'status': 'empty', 'progress': 0.0, 'assigned_arm': None}
            for _ in range(self.NUM_WIDGETS)
        ]
        
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # --- Unpack Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input ---
        # 1. Switch Active Arm (on press)
        if shift_held and not self.last_shift_held:
            self.active_arm_idx = 1 - self.active_arm_idx
            # SFX: UI_SWITCH

        # 2. Adjust Active Arm Parameters
        active_arm = self.arms[self.active_arm_idx]
        if not active_arm['target_idx']: # Can only adjust idle arms
            if movement == 1: # Up: Increase Speed
                active_arm['speed'] = min(5, active_arm['speed'] + 1)
            elif movement == 2: # Down: Decrease Speed
                active_arm['speed'] = max(1, active_arm['speed'] - 1)
            elif movement == 3: # Left: Decrease Precision
                active_arm['precision'] = max(70, active_arm['precision'] - 1)
            elif movement == 4: # Right: Increase Precision
                active_arm['precision'] = min(100, active_arm['precision'] + 1)

        # 3. Initiate Assembly (on press)
        if space_held and not self.last_space_held:
            if self.steps >= active_arm['busy_until']:
                target_idx = -1
                for i, widget in enumerate(self.widgets):
                    if widget['status'] in ['empty', 'failed']:
                        target_idx = i
                        break
                
                if target_idx != -1:
                    widget = self.widgets[target_idx]
                    
                    if widget['status'] == 'failed':
                        reward -= 1.0 # Reassembly penalty
                        # SFX: REASSEMBLE_START
                    else:
                        # SFX: ASSEMBLE_START
                        pass

                    widget['status'] = 'in_progress'
                    widget['progress'] = 0.0
                    widget['assigned_arm'] = self.active_arm_idx
                    
                    time_to_assemble_s = (6 - active_arm['speed']) * 0.75 # Speed 1=3.75s, 5=0.75s
                    steps_to_assemble = int(time_to_assemble_s * self.fps)
                    
                    active_arm['busy_until'] = self.steps + steps_to_assemble
                    active_arm['target_idx'] = target_idx

        self.last_space_held, self.last_shift_held = space_held, shift_held

        # --- Update Game Logic ---
        self.steps += 1
        self.time_remaining = max(0, self.time_remaining - 1.0 / self.fps)
        
        # Update widgets in progress
        for i, widget in enumerate(self.widgets):
            if widget['status'] == 'in_progress':
                arm_idx = widget['assigned_arm']
                arm = self.arms[arm_idx]
                
                time_to_assemble_s = (6 - arm['speed']) * 0.75
                steps_to_assemble = int(time_to_assemble_s * self.fps)
                
                old_progress_pct = int(widget['progress'] * 100)
                widget['progress'] += 1.0 / max(1, steps_to_assemble)
                new_progress_pct = int(widget['progress'] * 100)
                
                reward += (new_progress_pct - old_progress_pct) * 0.01 # +0.1 per 1%

                if widget['progress'] >= 1.0:
                    widget['progress'] = 1.0
                    is_success = self.np_random.random() < (arm['precision'] / 100.0)
                    
                    if is_success:
                        widget['status'] = 'complete'
                        self.completed_widgets += 1
                        reward += 3.0
                        self._add_particles(self.widget_positions[i], self.COLOR_SUCCESS, 30)
                        # SFX: SUCCESS
                    else:
                        widget['status'] = 'failed'
                        self._add_particles(self.widget_positions[i], self.COLOR_FAILURE, 20)
                        # SFX: FAILURE

                    # Free up the arm
                    self.arms[arm_idx]['target_idx'] = None
                    widget['assigned_arm'] = None

        # Update particles
        self.particles = [p for p in self.particles if p.update()]

        # --- Check Termination ---
        terminated = False
        if self.completed_widgets >= self.WIDGETS_TO_WIN:
            terminated = True
            reward += 100.0
            # SFX: GAME_WIN
        elif self.time_remaining <= 0:
            terminated = True
            reward -= 100.0
            # SFX: GAME_LOSE
        
        self.game_over = terminated
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _add_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append(Particle(pos.x, pos.y, color))
    
    def _render_game(self):
        # Draw conveyor belt
        pygame.draw.rect(self.screen, self.COLOR_BELT, self.belt_rect)
        
        # Draw widgets
        for i, widget in enumerate(self.widgets):
            pos = self.widget_positions[i]
            rect = pygame.Rect(pos.x - self.widget_size.x / 2, pos.y - self.widget_size.y / 2, self.widget_size.x, self.widget_size.y)
            
            if widget['status'] == 'empty':
                pygame.draw.rect(self.screen, self.COLOR_INACTIVE, rect, 2, border_radius=4)
            elif widget['status'] == 'failed':
                pygame.draw.rect(self.screen, self.COLOR_FAILURE, rect, 0, border_radius=4)
                fail_text = self.font_icon.render("!", True, self.COLOR_BG)
                self.screen.blit(fail_text, fail_text.get_rect(center=rect.center))
            elif widget['status'] == 'complete':
                pygame.draw.rect(self.screen, self.COLOR_SUCCESS, rect, 0, border_radius=4)
                # SFX: HUM_SUCCESS
            elif widget['status'] == 'in_progress':
                # Background
                pygame.draw.rect(self.screen, self.COLOR_INACTIVE, rect, 2, border_radius=4)
                # Progress bar
                progress_h = rect.height * widget['progress']
                progress_rect = pygame.Rect(rect.left, rect.bottom - progress_h, rect.width, progress_h)
                pygame.draw.rect(self.screen, self.COLOR_PROGRESS, progress_rect, 0, border_radius=4)
                # SFX: HUM_ASSEMBLY

        # Draw arms and beams
        for i, arm in enumerate(self.arms):
            base_pos = self.arm_positions[i]
            color = self.COLOR_ACTIVE if i == self.active_arm_idx else self.COLOR_INACTIVE
            pygame.draw.circle(self.screen, color, (int(base_pos.x), int(base_pos.y)), 15)
            pygame.draw.circle(self.screen, self.COLOR_BG, (int(base_pos.x), int(base_pos.y)), 10)
            
            if arm['target_idx'] is not None:
                target_pos = self.widget_positions[arm['target_idx']]
                # Draw a shimmering beam
                t = self.steps % self.fps
                for j in range(5):
                    alpha = 150 - j * 20 - (t * 5) % 50
                    if alpha > 0:
                        pygame.draw.line(self.screen, (*self.COLOR_PROGRESS, alpha), base_pos, target_pos, max(1, 5-j))

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Timer
        mins, secs = divmod(int(self.time_remaining), 60)
        timer_text = self.font_main.render(f"TIME: {mins:02}:{secs:02}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (20, 10))

        # Widget Count
        count_text = self.font_main.render(f"WIDGETS: {self.completed_widgets}/{self.NUM_WIDGETS}", True, self.COLOR_TEXT)
        self.screen.blit(count_text, count_text.get_rect(right=self.width - 20, top=10))

        # Arm Status
        for i, arm in enumerate(self.arms):
            is_active = i == self.active_arm_idx
            color = self.COLOR_ACTIVE if is_active else self.COLOR_INACTIVE
            
            x_offset = 20 if i == 0 else self.width - 270
            
            title = f"ARM {i+1}" + (" [ACTIVE]" if is_active else "")
            title_surf = self.font_small.render(title, True, color)
            self.screen.blit(title_surf, (x_offset, self.height - 70))
            
            # Speed bar
            speed_text = self.font_small.render("SPEED:", True, self.COLOR_TEXT)
            self.screen.blit(speed_text, (x_offset, self.height - 50))
            for j in range(5):
                bar_color = color if j < arm['speed'] else self.COLOR_INACTIVE
                pygame.draw.rect(self.screen, bar_color, (x_offset + 90 + j * 20, self.height - 48, 15, 10), border_radius=2)
            
            # Precision bar
            precision_text = self.font_small.render("PRECISION:", True, self.COLOR_TEXT)
            self.screen.blit(precision_text, (x_offset, self.height - 30))
            prec_val = (arm['precision'] - 70) / 30
            for j in range(10):
                bar_color = color if j / 10 < prec_val else self.COLOR_INACTIVE
                pygame.draw.rect(self.screen, bar_color, (x_offset + 90 + j * 10, self.height - 28, 8, 10), border_radius=2)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "completed_widgets": self.completed_widgets,
            "arm1_speed": self.arms[0]['speed'],
            "arm1_precision": self.arms[0]['precision'],
            "arm2_speed": self.arms[1]['speed'],
            "arm2_precision": self.arms[1]['precision'],
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Override screen for direct rendering
    pygame.display.set_caption("Widget Assembly Line")
    real_screen = pygame.display.set_mode((env.width, env.height))
    env.screen = real_screen

    terminated = False
    total_reward = 0
    
    # Game loop
    while not terminated:
        # --- Action Mapping for Human Player ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # --- Rendering ---
        # The observation is already the rendered frame, so we just need to display it
        # Note: The original code in __main__ was slightly incorrect for displaying
        # the observation. It should be transposed back for pygame's display.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Clock ---
        env.clock.tick(env.fps)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    print(f"Info: {info}")
    env.close()