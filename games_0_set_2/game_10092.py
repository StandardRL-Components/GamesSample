import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:47:10.579256
# Source Brief: brief_00092.md
# Brief Index: 92
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Grow a beautiful fractal by cloning orbs. Manage your energy to expand your creation and unlock new orb types."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select an orb. Press Shift to cycle the clone direction and Space to create a new orb."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    CLONE_DISTANCE = 60

    # --- Colors ---
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (5, 10, 20)
    COLOR_LINE = (100, 100, 120)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_ENERGY_BAR_BG = (40, 40, 60)
    COLOR_ENERGY_BAR_FG = (0, 255, 128)
    COLOR_ENERGY_BAR_WARN = (255, 100, 0)
    COLOR_SELECTION = (255, 255, 255)

    ORB_TYPES = [
        {"color": (0, 200, 255), "cost_mult": 1.0, "unlock_at": 0},   # Cyan
        {"color": (255, 100, 255), "cost_mult": 1.2, "unlock_at": 10}, # Magenta
        {"color": (255, 255, 0), "cost_mult": 1.5, "unlock_at": 25},  # Yellow
        {"color": (200, 255, 200), "cost_mult": 2.0, "unlock_at": 50}, # Mint
    ]

    class Orb:
        def __init__(self, id, pos, type_id, parent_id=None):
            self.id = id
            self.pos = pygame.Vector2(pos)
            self.type_id = type_id
            self.parent_id = parent_id
            self.radius = 10
            self.pulse_phase = random.uniform(0, 2 * math.pi)

    class Particle:
        def __init__(self, pos, vel, size, life, color):
            self.pos = pygame.Vector2(pos)
            self.vel = pygame.Vector2(vel)
            self.size = size
            self.life = life
            self.max_life = life
            self.color = color

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        self.orbs = {}
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.energy_pool = 0.0
        self.max_energy_pool = 100.0
        self.base_clone_cost = 5.0
        self.clone_cost_multiplier = 1.0
        self.next_orb_id = 0
        self.selected_orb_id = 0
        self.clone_direction = 0  # 0:up, 1:right, 2:down, 3:left
        self.unlocked_type_ids = set()
        self.size_milestones_achieved = set()
        self.prev_space_held = False
        self.prev_shift_held = False
        self.energy_warn_timer = 0

        self.reset()
        # self.validate_implementation() # Removed for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.energy_pool = self.max_energy_pool
        self.clone_cost_multiplier = 1.0
        self.next_orb_id = 0
        self.selected_orb_id = 0
        self.clone_direction = 0
        self.unlocked_type_ids = {0}
        self.size_milestones_achieved = set()
        self.prev_space_held = False
        self.prev_shift_held = False
        self.energy_warn_timer = 0
        self.particles.clear()
        self.orbs.clear()

        self._create_initial_orb()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        reward = self._handle_input(movement, space_pressed, shift_pressed)
        
        self._update_game_state()
        
        reward += self._check_unlocks_and_milestones()
        
        self.steps += 1
        terminated = self._check_termination()
        truncated = False
        if self.steps >= self.MAX_STEPS:
            terminated = True # Use terminated for time limit in this context
            
        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _create_initial_orb(self):
        initial_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        orb = self.Orb(self.next_orb_id, initial_pos, 0)
        self.orbs[self.next_orb_id] = orb
        self.next_orb_id += 1

    def _handle_input(self, movement, space_pressed, shift_pressed):
        reward = 0.0
        if shift_pressed:
            self.clone_direction = (self.clone_direction + 1) % 4
            # Sound: UI_TICK
        if movement > 0:
            self._select_relative_orb(movement)
        if space_pressed:
            reward += self._attempt_clone()
        return reward

    def _select_relative_orb(self, movement):
        if not self.orbs: return
        
        current_orb = self.orbs.get(self.selected_orb_id)
        if not current_orb: return
        
        direction_vectors = {
            1: pygame.Vector2(0, -1), # Up
            2: pygame.Vector2(0, 1),  # Down
            3: pygame.Vector2(-1, 0), # Left
            4: pygame.Vector2(1, 0),  # Right
        }
        move_vec = direction_vectors[movement]
        
        best_candidate = -1
        min_dist = float('inf')
        
        for orb_id, orb in self.orbs.items():
            if orb_id == self.selected_orb_id: continue
            
            to_orb_vec = orb.pos - current_orb.pos
            dist = to_orb_vec.length()
            
            if dist > 0:
                dot_product = move_vec.dot(to_orb_vec.normalize())
                if dot_product > 0.5: # In a 120-degree cone
                    if dist < min_dist:
                        min_dist = dist
                        best_candidate = orb_id
                        
        if best_candidate != -1:
            self.selected_orb_id = best_candidate
            # Sound: UI_SELECT

    def _attempt_clone(self):
        if not self.orbs: return 0.0
        parent_orb = self.orbs.get(self.selected_orb_id)
        if not parent_orb: return 0.0

        orb_type_info = self.ORB_TYPES[parent_orb.type_id]
        cost = (self.base_clone_cost * self.clone_cost_multiplier * orb_type_info["cost_mult"])

        if self.energy_pool >= cost:
            self.energy_pool -= cost
            
            direction_vectors = [
                pygame.Vector2(0, -1), pygame.Vector2(1, 0),
                pygame.Vector2(0, 1), pygame.Vector2(-1, 0)
            ]
            new_pos = parent_orb.pos + direction_vectors[self.clone_direction] * self.CLONE_DISTANCE

            new_orb = self.Orb(self.next_orb_id, new_pos, parent_orb.type_id, parent_orb.id)
            self.orbs[self.next_orb_id] = new_orb
            self.next_orb_id += 1
            
            self._spawn_particles(parent_orb.pos, new_pos, orb_type_info["color"])
            # Sound: CLONE_SUCCESS
            return 0.1 # Continuous reward for cloning
        else:
            self.energy_warn_timer = 15 # Flash energy bar for 15 frames
            # Sound: CLONE_FAIL
            return 0.0

    def _update_game_state(self):
        # Update clone cost difficulty
        if self.steps > 0 and self.steps % 200 == 0:
            self.clone_cost_multiplier += 0.1

        # Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.pos += p.vel
            p.life -= 1
            p.size = max(0, p.size - 0.2)
        
        if self.energy_warn_timer > 0:
            self.energy_warn_timer -= 1

    def _check_unlocks_and_milestones(self):
        reward = 0.0
        num_orbs = len(self.orbs)

        # Check for new orb type unlocks
        for i, orb_type in enumerate(self.ORB_TYPES):
            if i not in self.unlocked_type_ids and num_orbs >= orb_type["unlock_at"]:
                self.unlocked_type_ids.add(i)
                reward += 1.0
                # Sound: UNLOCK

        # Check for size milestones
        milestones = {10: 10, 25: 25, 50: 50}
        for size, r in milestones.items():
            if size not in self.size_milestones_achieved and num_orbs >= size:
                reward += r
                self.size_milestones_achieved.add(size)
        
        # Incremental rewards past 50
        if num_orbs > 50:
            tens_chunk = (num_orbs - 51) // 10
            for i in range(tens_chunk + 1):
                milestone = 60 + i * 10
                if milestone not in self.size_milestones_achieved:
                    reward += 10
                    self.size_milestones_achieved.add(milestone)
        
        self.score += reward
        return reward

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        
        min_possible_cost = float('inf')
        for orb_type_id in self.unlocked_type_ids:
            orb_type_info = self.ORB_TYPES[orb_type_id]
            cost = (self.base_clone_cost * self.clone_cost_multiplier * orb_type_info["cost_mult"])
            if cost < min_possible_cost:
                min_possible_cost = cost
        
        return self.energy_pool < min_possible_cost

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Draw connections
        for orb in self.orbs.values():
            if orb.parent_id is not None and orb.parent_id in self.orbs:
                parent_orb = self.orbs[orb.parent_id]
                pygame.draw.aaline(self.screen, self.COLOR_LINE, orb.pos, parent_orb.pos)

        # Draw orbs and selection highlight
        selected_orb = self.orbs.get(self.selected_orb_id)
        for orb in self.orbs.values():
            pulse = math.sin(self.steps * 0.1 + orb.pulse_phase)
            radius = int(orb.radius + pulse * 1.5)
            color = self.ORB_TYPES[orb.type_id]["color"]
            
            # Glow effect
            glow_radius = int(radius * 1.8)
            glow_alpha = int(70 + pulse * 20)
            self._draw_glowing_circle(orb.pos, glow_radius, color, glow_alpha)

            # Main orb
            pygame.gfxdraw.filled_circle(self.screen, int(orb.pos.x), int(orb.pos.y), radius, color)
            pygame.gfxdraw.aacircle(self.screen, int(orb.pos.x), int(orb.pos.y), radius, color)

            # Draw inner pattern based on type
            self._draw_orb_pattern(orb, radius)
            
        # Draw selection and direction indicator
        if selected_orb:
            # Selection ring
            select_pulse = abs(math.sin(self.steps * 0.2))
            select_radius = int(selected_orb.radius * 1.5 + 4)
            pygame.gfxdraw.aacircle(self.screen, int(selected_orb.pos.x), int(selected_orb.pos.y), select_radius, 
                                    (*self.COLOR_SELECTION, int(150 + select_pulse * 105)))
            
            # Direction arrow
            arrow_dist = select_radius + 5
            angle = self.clone_direction * math.pi / 2 - math.pi / 2
            p1 = selected_orb.pos + pygame.Vector2(arrow_dist, 0).rotate_rad(angle)
            p2 = selected_orb.pos + pygame.Vector2(arrow_dist - 5, -5).rotate_rad(angle)
            p3 = selected_orb.pos + pygame.Vector2(arrow_dist - 5, 5).rotate_rad(angle)
            pygame.draw.polygon(self.screen, self.COLOR_SELECTION, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)])

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            color = (*p.color, alpha)
            if p.size > 0:
                rect = pygame.Rect(p.pos.x - p.size/2, p.pos.y - p.size/2, p.size, p.size)
                pygame.draw.rect(self.screen, color, rect)

    def _draw_glowing_circle(self, pos, radius, color, alpha):
        temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*color, alpha), (radius, radius), radius)
        self.screen.blit(temp_surf, (int(pos.x - radius), int(pos.y - radius)), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_orb_pattern(self, orb, radius):
        color = (255, 255, 255)
        if orb.type_id == 1: # Inner circle
            pygame.gfxdraw.filled_circle(self.screen, int(orb.pos.x), int(orb.pos.y), int(radius * 0.5), color)
        elif orb.type_id == 2: # Triangle
            points = []
            for i in range(3):
                angle = 2 * math.pi * i / 3 - math.pi / 2
                points.append((orb.pos.x + math.cos(angle) * radius * 0.6, orb.pos.y + math.sin(angle) * radius * 0.6))
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif orb.type_id == 3: # Cross
            r = radius * 0.6
            pygame.draw.line(self.screen, color, (orb.pos.x - r, orb.pos.y), (orb.pos.x + r, orb.pos.y), 2)
            pygame.draw.line(self.screen, color, (orb.pos.x, orb.pos.y - r), (orb.pos.x, orb.pos.y + r), 2)

    def _render_ui(self):
        # Energy Bar
        bar_width = 250
        bar_height = 20
        bar_x = (self.SCREEN_WIDTH - bar_width) // 2
        bar_y = 10
        
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        
        energy_ratio = self.energy_pool / self.max_energy_pool
        fill_width = int(bar_width * energy_ratio)
        bar_color = self.COLOR_ENERGY_BAR_WARN if self.energy_warn_timer > 0 and self.energy_warn_timer % 4 < 2 else self.COLOR_ENERGY_BAR_FG
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, fill_width, bar_height), border_radius=4)
        
        # Score Text
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Steps Text
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 15, 10))

        # Fractal Size Text
        size_text = self.font_small.render(f"FRACTAL SIZE: {len(self.orbs)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(size_text, (self.SCREEN_WIDTH - size_text.get_width() - 15, 30))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "fractal_size": len(self.orbs)}

    def _spawn_particles(self, start_pos, end_pos, color):
        direction = (end_pos - start_pos).normalize()
        for _ in range(30):
            # Sound: PARTICLE_SPAWN
            offset = random.uniform(0, 1)
            pos = start_pos.lerp(end_pos, offset)
            
            vel_angle = direction.angle_to(pygame.Vector2(1,0)) + random.uniform(-45, 45)
            vel_mag = random.uniform(1, 3)
            vel = pygame.Vector2(vel_mag, 0).rotate(-vel_angle)
            
            life = random.randint(15, 30)
            size = random.uniform(2, 5)
            self.particles.append(self.Particle(pos, vel, size, life, color))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Create a display for human playing
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_caption("Fractal Orb Cloner")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    # Store previous action state for human play
    prev_action = [0, 0, 0]

    while running:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Space and Shift
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Only register movement/selection on key press, not hold
        if action[0] != 0 and prev_action[0] == action[0]:
            action[0] = 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Update previous action
        keys = pygame.key.get_pressed()
        prev_action[0] = 0
        if keys[pygame.K_UP]: prev_action[0] = 1
        elif keys[pygame.K_DOWN]: prev_action[0] = 2
        elif keys[pygame.K_LEFT]: prev_action[0] = 3
        elif keys[pygame.K_RIGHT]: prev_action[0] = 4

        # Render to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}, Steps: {info['steps']}")
            total_reward = 0
            obs, info = env.reset()
            # Add a small delay to notice the reset
            pygame.time.wait(1000)

        env.clock.tick(30) # Run at 30 FPS
        
    env.close()