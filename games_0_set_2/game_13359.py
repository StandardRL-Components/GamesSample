import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:59:09.255017
# Source Brief: brief_03359.md
# Brief Index: 3359
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player launches clay tablets to form ancient words.
    The goal is to activate all ancient machines by completing their corresponding word puzzles.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Launch clay tablets inscribed with ancient characters to complete words and reactivate mysterious machines."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to aim. Hold space to charge power and release to launch. "
        "Press shift to cycle through available tablets."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 2500

    # Colors
    COLOR_BG = (45, 35, 30)
    COLOR_SAND = (210, 180, 140)
    COLOR_SAND_DARK = (185, 155, 115)
    COLOR_PLAYER_AIM = (255, 100, 100)
    COLOR_TRAJECTORY = (255, 100, 100, 150)
    COLOR_POWER_BAR_BG = (80, 80, 80)
    COLOR_POWER_BAR_FILL = (100, 200, 255)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    COLOR_TABLET = (139, 69, 19)
    COLOR_TABLET_SHADOW = (0, 0, 0, 50)
    COLOR_TARGET_INACTIVE = (100, 90, 80)
    COLOR_TARGET_ACTIVE = (255, 215, 0)
    COLOR_MACHINE_INACTIVE = (120, 110, 100)
    COLOR_MACHINE_ACTIVE = (255, 215, 0)
    COLOR_SUCCESS_PARTICLE = (255, 215, 0)
    COLOR_FAIL_PARTICLE = (255, 50, 50)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 48)
        self.font_cuneiform = pygame.font.Font(None, 32)

        # --- Game State ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        # Player state
        self.aim_origin = pygame.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 50)
        self.aim_target = pygame.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.launch_power = 0.0
        
        # Action state tracking
        self.prev_space_state = 0
        self.prev_shift_state = 0
        
        # Game objects
        self.flying_tablets = []
        self.particles = []
        self.targets = []
        self.machines = []
        self.tablet_inventory = {}
        self.total_initial_tablets = 0

        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def _setup_level(self):
        """Initializes targets, machines, and tablet inventory for a new game."""
        self.targets.clear()
        self.machines.clear()
        self.tablet_inventory.clear()

        # Define level structure: (target_pos, machine_pos, word)
        level_data = [
            ((160, 200), (160, 100), "SUN"),
            ((480, 200), (480, 100), "WATER"),
            ((320, 300), (320, 50), "EARTH"),
        ]

        required_chars = set("".join(d[2] for d in level_data))

        for i, (t_pos, m_pos, word) in enumerate(level_data):
            machine = Machine(m_pos)
            self.machines.append(machine)
            self.targets.append(Target(t_pos, word, machine))

        # Populate inventory to guarantee a solvable puzzle
        for char in required_chars:
            self.tablet_inventory[char] = 3 # 3 of each required character
        # Add some extra random tablets
        for _ in range(5):
            char = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            if char not in self.tablet_inventory:
                self.tablet_inventory[char] = 0
            self.tablet_inventory[char] += 1
            
        self.current_tablet_char = sorted(self.tablet_inventory.keys())[0]
        self.total_initial_tablets = sum(self.tablet_inventory.values())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.aim_target = pygame.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.launch_power = 0.0
        self.prev_space_state = 0
        self.prev_shift_state = 0

        self.flying_tablets.clear()
        self.particles.clear()
        
        self._setup_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_action, shift_action = action
        reward = 0.0

        self._handle_input(movement, space_action, shift_action)
        
        reward += self._update_physics()
        
        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()

        if terminated:
            if all(m.is_active for m in self.machines):
                reward += 100.0 # Victory bonus
            else:
                reward -= 100.0 # Failure penalty
            self.score += reward
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, movement, space_action, shift_action):
        # --- Aiming ---
        aim_speed = 4.0
        if movement == 1: self.aim_target.y -= aim_speed # Up
        elif movement == 2: self.aim_target.y += aim_speed # Down
        elif movement == 3: self.aim_target.x -= aim_speed # Left
        elif movement == 4: self.aim_target.x += aim_speed # Right
        
        # Clamp aim target to screen
        self.aim_target.x = np.clip(self.aim_target.x, 0, self.SCREEN_WIDTH)
        self.aim_target.y = np.clip(self.aim_target.y, 0, self.SCREEN_HEIGHT - 100)

        # --- Tablet Selection (on press) ---
        if shift_action == 1 and self.prev_shift_state == 0:
            # `// Sound: UI click`
            available_chars = sorted(self.tablet_inventory.keys())
            if not available_chars: return
            try:
                current_idx = available_chars.index(self.current_tablet_char)
                next_idx = (current_idx + 1) % len(available_chars)
                self.current_tablet_char = available_chars[next_idx]
            except ValueError:
                self.current_tablet_char = available_chars[0]

        # --- Power Charging ---
        if space_action == 1:
            self.launch_power = min(1.0, self.launch_power + 0.04)
        
        # --- Launching (on release) ---
        if space_action == 0 and self.prev_space_state == 1:
            if self.launch_power > 0.1 and self.tablet_inventory.get(self.current_tablet_char, 0) > 0:
                # `// Sound: Tablet launch swoosh`
                self.tablet_inventory[self.current_tablet_char] -= 1
                
                direction = (self.aim_target - self.aim_origin).normalize()
                speed = 2.0 + self.launch_power * 10.0
                velocity = pygame.Vector3(direction.x * speed, direction.y * speed * 0.5, 6.0 + self.launch_power * 4)
                
                self.flying_tablets.append(FlyingTablet(
                    pos=pygame.Vector3(self.aim_origin.x, self.aim_origin.y, 0),
                    vel=velocity,
                    char=self.current_tablet_char
                ))
            self.launch_power = 0.0

        self.prev_space_state = space_action
        self.prev_shift_state = shift_action

    def _update_physics(self):
        reward = 0.0
        
        # Update particles
        for p in self.particles[:]:
            p.update()
            if not p.is_alive:
                self.particles.remove(p)

        # Update flying tablets
        for tablet in self.flying_tablets[:]:
            tablet.update()
            
            # Check for impact with ground
            if tablet.pos.z <= 0:
                tablet.pos.z = 0
                # `// Sound: Tablet shatter on ground`
                self._create_particles(tablet.pos, self.COLOR_FAIL_PARTICLE, 15)
                self.flying_tablets.remove(tablet)
                continue

            # Check for impact with targets
            for target in self.targets:
                if not target.is_complete and tablet.pos.z < 10 and target.is_colliding(tablet.pos):
                    # `// Sound: Tablet shatter on target`
                    hit_reward = target.register_hit(tablet.char)
                    reward += hit_reward
                    
                    if hit_reward > 0:
                        self._create_particles(tablet.pos, self.COLOR_SUCCESS_PARTICLE, 30, is_glow=True)
                        if hit_reward > 1.0: # Word complete or machine activated
                             # `// Sound: Success chime`
                             pass
                    else:
                        self._create_particles(tablet.pos, self.COLOR_FAIL_PARTICLE, 15)
                    
                    self.flying_tablets.remove(tablet)
                    break # Tablet can only hit one target
            
            # Remove if out of bounds
            if not (0 < tablet.pos.x < self.SCREEN_WIDTH and 0 < tablet.pos.y < self.SCREEN_HEIGHT * 2):
                self.flying_tablets.remove(tablet)

        return reward

    def _check_termination(self):
        # Victory condition
        if all(m.is_active for m in self.machines):
            return True
        
        # Failure condition
        tablets_left = sum(self.tablet_inventory.values())
        if tablets_left <= 0 and not self.flying_tablets:
            return True
            
        # Max steps
        if self.steps >= self.MAX_STEPS:
            return True
            
        return False

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
            "tablets_remaining": sum(self.tablet_inventory.values()),
            "machines_activated": sum(1 for m in self.machines if m.is_active),
        }

    def _render_game(self):
        # Render sand pit
        pygame.draw.rect(self.screen, self.COLOR_SAND_DARK, (20, 20, self.SCREEN_WIDTH - 40, self.SCREEN_HEIGHT - 40))
        pygame.draw.rect(self.screen, self.COLOR_SAND, (25, 25, self.SCREEN_WIDTH - 50, self.SCREEN_HEIGHT - 55))

        # Render machines and targets
        for machine in self.machines:
            machine.draw(self.screen)
        for target in self.targets:
            target.draw(self.screen, self.font_cuneiform)
            
        # Render objects sorted by Y-position for correct layering
        render_queue = self.flying_tablets[:]
        render_queue.sort(key=lambda obj: obj.pos.y)

        for tablet in render_queue:
            tablet.draw(self.screen, self.font_cuneiform)
            
        # Render particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # --- Aiming UI ---
        if not self.flying_tablets and not self.game_over:
            # Trajectory line
            self._draw_trajectory()
            # Aim reticle
            pygame.draw.circle(self.screen, self.COLOR_PLAYER_AIM, (int(self.aim_target.x), int(self.aim_target.y)), 10, 2)
            pygame.draw.line(self.screen, self.COLOR_PLAYER_AIM, (self.aim_target.x - 15, self.aim_target.y), (self.aim_target.x + 15, self.aim_target.y), 2)
            pygame.draw.line(self.screen, self.COLOR_PLAYER_AIM, (self.aim_target.x, self.aim_target.y - 15), (self.aim_target.x, self.aim_target.y + 15), 2)
            
            # Power bar
            power_bar_width = 100
            power_bar_height = 20
            power_bar_x = self.aim_origin.x - power_bar_width / 2
            power_bar_y = self.aim_origin.y + 20
            fill_width = self.launch_power * power_bar_width
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_BG, (power_bar_x, power_bar_y, power_bar_width, power_bar_height))
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_FILL, (power_bar_x, power_bar_y, fill_width, power_bar_height))
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (power_bar_x, power_bar_y, power_bar_width, power_bar_height), 1)

        # --- Text Info ---
        score_text = f"Score: {self.score:.1f}"
        self._draw_text(score_text, (10, 10), self.font_small)

        tablets_left = sum(self.tablet_inventory.values())
        tablets_text = f"Tablets Left: {tablets_left}"
        self._draw_text(tablets_text, (self.SCREEN_WIDTH - 150, 10), self.font_small)
        
        # --- Inventory ---
        self._draw_inventory()
        
        # --- Game Over Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "CITY RESTORED" if all(m.is_active for m in self.machines) else "FAILURE"
            color = self.COLOR_MACHINE_ACTIVE if all(m.is_active for m in self.machines) else self.COLOR_FAIL_PARTICLE
            self._draw_text(msg, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20), self.font_large, color=color, center=True)
            self._draw_text(f"Final Score: {self.score:.1f}", (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30), self.font_medium, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _draw_inventory(self):
        start_x = self.SCREEN_WIDTH // 2 - 100
        y = self.SCREEN_HEIGHT - 35
        
        available_chars = sorted(self.tablet_inventory.keys())
        if not available_chars: return
        
        try:
            current_idx = available_chars.index(self.current_tablet_char)
        except ValueError:
            current_idx = 0
            if available_chars:
                self.current_tablet_char = available_chars[0]
            else:
                return # No tablets left to display

        display_indices = [(current_idx + i) % len(available_chars) for i in range(-2, 3)]
        
        for i, list_idx in enumerate(display_indices):
            char = available_chars[list_idx]
            count = self.tablet_inventory[char]
            
            pos_x = start_x + i * 50
            
            is_current = (list_idx == current_idx)
            size = 30 if is_current else 20
            color = self.COLOR_TABLET if count > 0 else (80, 40, 10)
            
            rect = pygame.Rect(pos_x - size/2, y - size/2, size, size)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            if is_current:
                 pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_FILL, rect, 2, border_radius=4)

            if count > 0:
                self._draw_text(char, rect.center, self.font_medium, center=True)
                self._draw_text(str(count), (rect.centerx, rect.bottom + 5), self.font_small, center=True)

    def _draw_trajectory(self):
        if self.launch_power > 0.05:
            direction = (self.aim_target - self.aim_origin).normalize()
            speed = 2.0 + self.launch_power * 10.0
            vel = pygame.Vector3(direction.x * speed, direction.y * speed * 0.5, 6.0 + self.launch_power * 4)
            pos = pygame.Vector3(self.aim_origin.x, self.aim_origin.y, 0)
            
            points = []
            for _ in range(20): # Simulate 20 steps of the trajectory
                vel.z -= FlyingTablet.GRAVITY
                pos += vel * 0.5 # Scale down for preview
                if pos.z < 0: break
                
                # Simple perspective scaling for the dot
                screen_x, screen_y = pos.x, pos.y + pos.z * 0.5
                points.append((screen_x, screen_y))

            if len(points) > 1:
                pygame.draw.lines(self.screen, self.COLOR_TRAJECTORY, False, points, 2)


    def _create_particles(self, pos, color, count, is_glow=False):
        for _ in range(count):
            self.particles.append(Particle(pos, color, is_glow))

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Helper Classes ---

class FlyingTablet:
    GRAVITY = 0.2
    
    def __init__(self, pos, vel, char):
        self.pos = pos # pygame.Vector3
        self.vel = vel # pygame.Vector3
        self.char = char
        self.angle = 0
        self.rot_speed = random.uniform(-5, 5)

    def update(self):
        self.vel.z -= self.GRAVITY
        self.pos += self.vel
        self.angle += self.rot_speed

    def draw(self, surface, font):
        # Project 3D position to 2D screen space
        screen_x = int(self.pos.x)
        screen_y = int(self.pos.y + self.pos.z * 0.5) # Simple perspective
        
        # Shadow
        shadow_y = int(self.pos.y)
        shadow_size = max(5, 20 - self.pos.z * 0.2)
        shadow_alpha = max(10, 80 - self.pos.z * 1.0)
        shadow_surf = pygame.Surface((shadow_size*2, shadow_size*2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0,0,0,shadow_alpha), (0,0,shadow_size*2, shadow_size))
        surface.blit(shadow_surf, (screen_x - shadow_size, shadow_y - shadow_size/2))
        
        # Tablet
        size = 20
        rect = pygame.Rect(screen_x - size/2, screen_y - size/2, size, size)
        pygame.draw.rect(surface, GameEnv.COLOR_TABLET, rect, border_radius=3)
        
        text_surf = font.render(self.char, True, GameEnv.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=rect.center)
        surface.blit(text_surf, text_rect)

class Target:
    def __init__(self, pos, word, machine):
        self.pos = pygame.Vector2(pos)
        self.word = word
        self.machine = machine
        self.width = len(word) * 25 + 10
        self.height = 40
        self.progress = ""
    
    @property
    def is_complete(self):
        return self.progress == self.word

    def is_colliding(self, tablet_pos_3d):
        rect = pygame.Rect(self.pos.x - self.width / 2, self.pos.y - self.height / 2, self.width, self.height)
        return rect.collidepoint(tablet_pos_3d.x, tablet_pos_3d.y)

    def register_hit(self, char):
        if self.is_complete:
            return 0.0

        next_char_needed = self.word[len(self.progress)]
        if char == next_char_needed:
            self.progress += char
            reward = 0.1 # Correct character
            if self.is_complete:
                reward += 5.0 # Word complete
                if not self.machine.is_active:
                    self.machine.activate()
                    reward += 10.0 # Machine activated
            return reward
        return 0.0 # Incorrect character

    def draw(self, surface, font):
        rect = pygame.Rect(self.pos.x - self.width / 2, self.pos.y - self.height / 2, self.width, self.height)
        color = GameEnv.COLOR_TARGET_ACTIVE if self.is_complete else GameEnv.COLOR_TARGET_INACTIVE
        pygame.draw.rect(surface, color, rect, 2, border_radius=5)
        
        for i, char in enumerate(self.word):
            char_pos_x = rect.left + 15 + i * 25
            char_pos_y = rect.centery
            
            char_color = GameEnv.COLOR_TEXT
            if i < len(self.progress):
                char_color = GameEnv.COLOR_SUCCESS_PARTICLE
            
            text_surf = font.render(char, True, char_color)
            text_rect = text_surf.get_rect(center=(char_pos_x, char_pos_y))
            surface.blit(text_surf, text_rect)

class Machine:
    def __init__(self, pos):
        self.pos = pygame.Vector2(pos)
        self.is_active = False
        self.activation_glow = 0.0

    def activate(self):
        self.is_active = True
        self.activation_glow = 1.0

    def draw(self, surface):
        if self.activation_glow > 0:
            self.activation_glow -= 0.02

        color = GameEnv.COLOR_MACHINE_ACTIVE if self.is_active else GameEnv.COLOR_MACHINE_INACTIVE
        base_rect = pygame.Rect(self.pos.x - 20, self.pos.y - 15, 40, 30)
        top_rect = pygame.Rect(self.pos.x - 15, self.pos.y - 25, 30, 10)
        
        pygame.draw.rect(surface, color, base_rect, border_radius=4)
        pygame.draw.rect(surface, color, top_rect, border_radius=4)
        
        if self.activation_glow > 0:
            radius = (1.0 - self.activation_glow) * 40
            alpha = self.activation_glow * 150
            glow_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*GameEnv.COLOR_MACHINE_ACTIVE, alpha), (radius, radius), radius)
            surface.blit(glow_surf, (self.pos.x - radius, self.pos.y - radius), special_flags=pygame.BLEND_RGBA_ADD)

class Particle:
    def __init__(self, pos, color, is_glow=False):
        self.pos = pygame.Vector3(pos)
        self.is_glow = is_glow
        self.vel = pygame.Vector3(
            random.uniform(-2, 2), 
            random.uniform(-2, 2), 
            random.uniform(1, 4)
        )
        self.lifespan = random.randint(20, 40)
        self.color = color
        self.is_alive = True

    def update(self):
        self.lifespan -= 1
        if self.lifespan <= 0:
            self.is_alive = False
            return
        
        self.vel.z -= 0.2 # Gravity
        self.pos += self.vel

    def draw(self, surface):
        if not self.is_alive: return
        
        screen_x = int(self.pos.x)
        screen_y = int(self.pos.y + self.pos.z * 0.5)
        
        alpha = int(255 * (self.lifespan / 40))
        color = (*self.color, alpha)
        
        size = 3 if self.is_glow else 2
        
        if self.is_glow:
            glow_surf = pygame.Surface((size*4, size*4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, color, (size*2, size*2), size*2)
            surface.blit(glow_surf, (screen_x - size*2, screen_y - size*2), special_flags=pygame.BLEND_RGBA_ADD)
        else:
            pygame.draw.circle(surface, self.color, (screen_x, screen_y), size)


if __name__ == '__main__':
    # --- Example Usage ---
    env = GameEnv(render_mode="rgb_array")
    
    # Manual play loop
    # Set the video driver to a real one for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Sumerian Tablet Smasher")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Mapping keyboard keys to MultiDiscrete actions
    key_map = {
        pygame.K_w: 1, pygame.K_UP: 1,
        pygame.K_s: 2, pygame.K_DOWN: 2,
        pygame.K_a: 3, pygame.K_LEFT: 3,
        pygame.K_d: 4, pygame.K_RIGHT: 4,
    }

    running = True
    while running:
        # --- Create action from keyboard input ---
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
        
        if not env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a surface to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()