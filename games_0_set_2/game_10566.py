import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:40:23.118233
# Source Brief: brief_00566.md
# Brief Index: 566
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game objects to keep the main class cleaner
class Nucleus:
    def __init__(self, pos, radius, hp, color_id, color_rgb):
        self.pos = np.array(pos, dtype=float)
        self.radius = radius
        self.hp = hp
        self.max_hp = hp
        self.color_id = color_id
        self.color_rgb = color_rgb
        self.hit_flash_timer = 0

    def take_damage(self, damage):
        self.hp = max(0, self.hp - damage)
        self.hit_flash_timer = 5  # Flash for 5 frames

class Projectile:
    def __init__(self, pos, vel, color_id, color_rgb, p_type="normal"):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.color_id = color_id
        self.color_rgb = color_rgb
        self.type = p_type
        self.trail = []
        self.age = 0

class Particle:
    def __init__(self, pos, vel, color, lifespan):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Fire specialized arrows from your bow to destroy atomic nuclei. Match arrow types to nuclei "
        "colors for bonus damage and clear each level before you run out of shots."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to adjust power and ↑↓ to aim. Press space to shoot and shift to "
        "cycle arrow types."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    GRAVITY = 0.15

    # --- COLORS ---
    COLOR_BG = (15, 18, 42)
    COLOR_BG_STARS = (100, 100, 120)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_AIM_LINE = (255, 255, 255, 100)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (40, 45, 80, 150)
    NUCLEUS_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 80, 255)    # Blue
    ]
    ARROW_TYPE_COLORS = {
        "Normal": (255, 255, 0),
        "Split": (0, 255, 255),
        "Pierce": (255, 0, 255)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- GYM SPACES ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- PYGAME SETUP ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # --- GAME STATE (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.nuclei = []
        self.projectiles = []
        self.particles = []
        self.stars = []
        
        # Player state
        self.archer_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 20)
        self.aim_angle = 90.0  # Degrees
        self.aim_power = 50.0  # Velocity magnitude
        self.arrows_left = 0
        self.arrow_types = ["Normal"]
        self.current_arrow_type_idx = 0
        
        # Action handling state
        self.space_was_held = False
        self.shift_was_held = False

        self._generate_stars()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.arrow_types = ["Normal"]
        self.current_arrow_type_idx = 0
        self._setup_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        # --- ACTION HANDLING ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Adjust aim
        if movement == 1: self.aim_angle = min(170, self.aim_angle + 1.5) # Up
        elif movement == 2: self.aim_angle = max(10, self.aim_angle - 1.5) # Down
        elif movement == 3: self.aim_power = max(20, self.aim_power - 2) # Left
        elif movement == 4: self.aim_power = min(100, self.aim_power + 2) # Right

        # Cycle arrow type (on press, not hold)
        if shift_held and not self.shift_was_held:
            self.current_arrow_type_idx = (self.current_arrow_type_idx + 1) % len(self.arrow_types)
        self.shift_was_held = shift_held

        # Shoot arrow (on press, not hold)
        if space_held and not self.space_was_held and self.arrows_left > 0:
            self._shoot_arrow()
        self.space_was_held = space_held

        # --- GAME LOGIC UPDATE ---
        reward += self._update_projectiles()
        self._update_particles()
        
        for n in self.nuclei:
            if n.hit_flash_timer > 0:
                n.hit_flash_timer -= 1

        # --- CHECK TERMINATION ---
        terminated = self._check_termination()
        if not terminated and not self.nuclei: # Win condition
            reward += 100
            self.score += 100
            self.level += 1
            self._setup_level() # Progress to next level
            terminated = False # Continue playing

        self.score += reward
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
        
    def _setup_level(self):
        self.nuclei.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.game_over = False
        
        # Unlock new arrows based on level
        if self.level >= 3 and "Split" not in self.arrow_types: self.arrow_types.append("Split")
        if self.level >= 6 and "Pierce" not in self.arrow_types: self.arrow_types.append("Pierce")

        num_nuclei = 3 + (self.level - 1) // 2
        nucleus_hp = 10 + (self.level - 1) * 5
        self.arrows_left = num_nuclei * 4
        
        for _ in range(num_nuclei):
            placed = False
            while not placed:
                pos = (self.np_random.integers(100, self.SCREEN_WIDTH - 100),
                       self.np_random.integers(50, self.SCREEN_HEIGHT - 150))
                radius = self.np_random.integers(15, 25)
                
                # Check for overlap with existing nuclei
                overlap = False
                for n in self.nuclei:
                    dist = math.hypot(pos[0] - n.pos[0], pos[1] - n.pos[1])
                    if dist < radius + n.radius + 10:
                        overlap = True
                        break
                if not overlap:
                    color_id = self.np_random.integers(0, len(self.NUCLEUS_COLORS))
                    self.nuclei.append(Nucleus(pos, radius, nucleus_hp, color_id, self.NUCLEUS_COLORS[color_id]))
                    placed = True

    def _shoot_arrow(self):
        self.arrows_left -= 1
        angle_rad = math.radians(self.aim_angle)
        vel_x = self.aim_power / 10 * math.cos(angle_rad)
        vel_y = -self.aim_power / 10 * math.sin(angle_rad)
        
        arrow_type = self.arrow_types[self.current_arrow_type_idx]
        color_id = self.current_arrow_type_idx % len(self.NUCLEUS_COLORS) # Simple mapping for now
        color_rgb = self.ARROW_TYPE_COLORS[arrow_type]
        
        self.projectiles.append(Projectile(self.archer_pos, (vel_x, vel_y), color_id, color_rgb, arrow_type))
        
    def _update_projectiles(self):
        reward = 0
        projectiles_to_add = []
        
        for p in self.projectiles[:]:
            p.vel[1] += self.GRAVITY
            p.pos += p.vel
            p.age += 1
            
            # Add trail particles
            if p.age % 2 == 0:
                p.trail.append(p.pos.copy())
                if len(p.trail) > 10:
                    p.trail.pop(0)

            # Check for collision with nuclei
            hit = False
            for n in self.nuclei[:]:
                dist = math.hypot(p.pos[0] - n.pos[0], p.pos[1] - n.pos[1])
                if dist < n.radius:
                    hit = True
                    
                    # Calculate damage and rewards
                    damage = 10
                    reward += 1
                    if p.color_id == n.color_id:
                        damage *= 2
                        reward += 2
                    
                    n.take_damage(damage)
                    self._create_explosion(n.pos, n.color_rgb, 20)
                    
                    if n.hp <= 0:
                        reward += 10
                        self._create_explosion(n.pos, n.color_rgb, 50)
                        self.nuclei.remove(n)
                    
                    # Handle arrow type specific logic
                    if p.type == "Split" and p.age > 5: # Split after a short delay
                        for i in range(-1, 2, 2):
                            angle = math.atan2(-p.vel[1], p.vel[0]) + math.radians(20 * i)
                            speed = math.hypot(p.vel[0], p.vel[1]) * 0.8
                            new_vel = (math.cos(angle) * speed, -math.sin(angle) * speed)
                            projectiles_to_add.append(Projectile(p.pos, new_vel, p.color_id, p.color_rgb, "Normal"))
                        
                    if p.type != "Pierce":
                        if p in self.projectiles:
                           self.projectiles.remove(p)
                    break # Projectile hit a nucleus
            
            # Remove projectile if it goes off-screen
            if not (0 < p.pos[0] < self.SCREEN_WIDTH and 0 < p.pos[1] < self.SCREEN_HEIGHT + 50):
                if p in self.projectiles:
                    self.projectiles.remove(p)
        
        self.projectiles.extend(projectiles_to_add)
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p.pos += p.vel
            p.lifespan -= 1
            if p.lifespan <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append(Particle(pos.copy(), vel, color, lifespan))

    def _check_termination(self):
        if self.arrows_left <= 0 and not self.projectiles:
            self.game_over = True
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
            "level": self.level,
            "arrows_left": self.arrows_left,
            "nuclei_left": len(self.nuclei)
        }

    def _generate_stars(self):
        self.stars = []
        for _ in range(100):
            pos = (self.np_random.integers(0, self.SCREEN_WIDTH),
                   self.np_random.integers(0, self.SCREEN_HEIGHT))
            size = self.np_random.integers(1, 3)
            self.stars.append((pos, size))

    def _render_game(self):
        # Draw stars
        for pos, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_BG_STARS, (pos[0], pos[1], size, size))

        # Draw nuclei
        for n in self.nuclei:
            pos_int = (int(n.pos[0]), int(n.pos[1]))
            # Health ring (background)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], n.radius + 2, (50, 50, 70))
            # Health ring (foreground)
            health_angle = 360 * (n.hp / n.max_hp)
            if health_angle > 0:
                rect = (pos_int[0] - n.radius - 2, pos_int[1] - n.radius - 2, (n.radius + 2)*2, (n.radius + 2)*2)
                pygame.draw.arc(self.screen, n.color_rgb, rect, math.radians(90), math.radians(90 + health_angle), 3)

            # Main body
            color = (255, 255, 255) if n.hit_flash_timer > 0 else n.color_rgb
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], n.radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], n.radius, color)

        # Draw projectiles
        for p in self.projectiles:
            # Draw trail
            if len(p.trail) > 1:
                trail_color = (*p.color_rgb, 150)
                pygame.draw.aalines(self.screen, trail_color, False, p.trail, 1)
            # Draw projectile head
            pos_int = (int(p.pos[0]), int(p.pos[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 3, p.color_rgb)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 3, p.color_rgb)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p.lifespan / p.max_lifespan))
            color = (*p.color, alpha)
            s = pygame.Surface((2, 2), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p.pos[0]), int(p.pos[1])))

        # Draw archer platform
        pygame.draw.arc(self.screen, self.COLOR_PLAYER, (self.archer_pos[0]-20, self.archer_pos[1]-15, 40, 30), 0, math.pi, 3)

        # Draw aiming line
        if not self.game_over:
            angle_rad = math.radians(self.aim_angle)
            vel_x = self.aim_power / 10 * math.cos(angle_rad)
            vel_y = -self.aim_power / 10 * math.sin(angle_rad)
            path = []
            pos = np.array(self.archer_pos, dtype=float)
            vel = np.array([vel_x, vel_y], dtype=float)
            for _ in range(20):
                vel[1] += self.GRAVITY * 2
                pos += vel * 2
                path.append(pos.copy())
            if len(path) > 1:
                pygame.draw.lines(self.screen, self.COLOR_AIM_LINE, False, path, 1)

    def _render_ui(self):
        # --- Draw UI background panels ---
        s = pygame.Surface((180, 80), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (5, 5)) # Top-left for score/level
        
        s2 = pygame.Surface((200, 80), pygame.SRCALPHA)
        s2.fill(self.COLOR_UI_BG)
        self.screen.blit(s2, (self.SCREEN_WIDTH - 205, self.SCREEN_HEIGHT - 85)) # Bottom-right for arrow info

        # --- Text Rendering ---
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 15))
        
        level_text = self.font_small.render(f"Level: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (15, 50))

        arrow_text = self.font_large.render(f"Arrows: {self.arrows_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(arrow_text, (self.SCREEN_WIDTH - 195, self.SCREEN_HEIGHT - 75))

        # --- Current Arrow Type Display ---
        current_type_str = self.arrow_types[self.current_arrow_type_idx]
        type_color = self.ARROW_TYPE_COLORS[current_type_str]
        type_text = self.font_small.render(f"Type: {current_type_str}", True, type_color)
        self.screen.blit(type_text, (self.SCREEN_WIDTH - 195, self.SCREEN_HEIGHT - 45))

    def close(self):
        pygame.font.quit()
        pygame.quit()

# Example usage to run and visualize the environment
if __name__ == '__main__':
    # This block will not run in the testing environment, but is useful for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Controls ---
    # Arrow Keys: Aim
    # Spacebar: Shoot
    # Left Shift: Cycle Arrow Type
    
    obs, info = env.reset()
    done = False
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Atomic Archery")
    human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    action = [0, 0, 0] # Start with no-op
    
    while not done:
        # Map pygame events to gymnasium action
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()