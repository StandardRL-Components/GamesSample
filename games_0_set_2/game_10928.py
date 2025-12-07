import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:10:31.455634
# Source Brief: brief_00928.md
# Brief Index: 928
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes ---

class Particle:
    """A single particle for effects."""
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.radius -= 0.1
        self.vel *= 0.98 # Damping

    def draw(self, surface):
        if self.radius > 0:
            # Fade out effect
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            alpha = max(0, min(255, alpha))
            
            # Use gfxdraw for anti-aliased filled circles
            pygame.gfxdraw.filled_circle(
                surface, int(self.pos.x), int(self.pos.y), int(self.radius), 
                (*self.color, alpha)
            )
            pygame.gfxdraw.aacircle(
                surface, int(self.pos.x), int(self.pos.y), int(self.radius), 
                (*self.color, alpha)
            )

class Target:
    """A moving target for the player to hit."""
    def __init__(self, screen_width, screen_height, base_speed):
        self.radius = random.randint(15, 25)
        speed_multiplier = base_speed + random.uniform(-0.2, 0.2)
        
        # Spawn on left or right edge
        if random.choice([True, False]):
            self.pos = pygame.Vector2(-self.radius, random.uniform(self.radius, screen_height * 0.7))
            self.vel = pygame.Vector2(random.uniform(0.8, 1.2) * speed_multiplier, random.uniform(-0.3, 0.3) * speed_multiplier)
        else:
            self.pos = pygame.Vector2(screen_width + self.radius, random.uniform(self.radius, screen_height * 0.7))
            self.vel = pygame.Vector2(random.uniform(-1.2, -0.8) * speed_multiplier, random.uniform(-0.3, 0.3) * speed_multiplier)
            
        self.color = (0, 255, 255) # Cyan

    def update(self):
        self.pos += self.vel

    def draw(self, surface):
        # Glow effect
        for i in range(4, 0, -1):
            alpha = 80 - i * 20
            pygame.gfxdraw.filled_circle(
                surface, int(self.pos.x), int(self.pos.y), self.radius + i, 
                (*self.color, alpha)
            )
        # Main circle
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)

class FiredLine:
    """The projectile fired by the player."""
    def __init__(self, start_pos, angle, is_wide, grid_phase):
        self.start_pos = pygame.Vector2(start_pos)
        self.pos = pygame.Vector2(start_pos)
        self.angle = angle
        self.is_wide = is_wide
        self.speed = 25
        self.vel = pygame.Vector2(math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle))) * self.speed
        self.trail = [self.pos.copy()]
        self.max_trail_length = 50
        self.active = True
        self.grid_phase = grid_phase
        self.grid_amplitude = 25

    def update(self, current_steps):
        if not self.active:
            return
            
        original_pos = self.pos + self.vel
        
        # Apply grid oscillation
        oscillation = math.sin((current_steps + self.grid_phase) * 0.05) * self.grid_amplitude
        self.pos = pygame.Vector2(original_pos.x + oscillation, original_pos.y)

        self.trail.append(self.pos.copy())
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)

    def draw(self, surface):
        color = (255, 0, 255) # Magenta
        if len(self.trail) > 1:
            # Main line
            pygame.draw.aalines(surface, color, False, [(int(p.x), int(p.y)) for p in self.trail], 1)
            
            if self.is_wide:
                # Draw two parallel lines for wide mode
                offset_angle = self.angle + 90
                offset_vec = pygame.Vector2(math.cos(math.radians(offset_angle)), math.sin(math.radians(offset_angle))) * 4
                
                trail1 = [(int(p.x + offset_vec.x), int(p.y + offset_vec.y)) for p in self.trail]
                trail2 = [(int(p.x - offset_vec.x), int(p.y - offset_vec.y)) for p in self.trail]
                
                pygame.draw.aalines(surface, color, False, trail1, 1)
                pygame.draw.aalines(surface, color, False, trail2, 1)


# --- Gymnasium Environment ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Aim a powerful laser cannon to shoot down incoming targets. Switch between a precise beam and a wide shot to maximize your score."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to aim. Press space to fire and shift to toggle between narrow and wide shots."
    )
    auto_advance = True

    # --- Colors and Constants ---
    COLOR_BG = (0, 5, 20)
    COLOR_GRID = (20, 30, 80)
    COLOR_PLAYER = (255, 0, 255)
    COLOR_HIT = (255, 255, 0)
    COLOR_UI_SCORE = (255, 255, 255)
    COLOR_UI_MISS = (255, 50, 50)
    
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1500
    SCORE_TO_WIN = 80
    MAX_MISSES = 3

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
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.line_angle = 0.0
        self.is_wide_mode = False
        self.targets = []
        self.fired_lines = []
        self.particles = []
        self.target_spawn_timer = 0
        self.base_target_speed = 0.0
        self.previous_space_held = False
        self.previous_shift_held = False
        
        # self.reset() is called by the environment wrapper
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        
        self.line_angle = 270.0  # Straight up
        self.is_wide_mode = False
        
        self.targets = []
        self.fired_lines = []
        self.particles = []
        
        self.target_spawn_timer = 0
        self.base_target_speed = 1.0
        
        self.previous_space_held = False
        self.previous_shift_held = False

        # Spawn initial targets
        for _ in range(3):
            self._spawn_target()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            reward = 0
            terminated = True
            truncated = self.steps >= self.MAX_STEPS
            return self._get_observation(), reward, terminated, truncated, self._get_info()

        movement, space_action, shift_action = action
        space_held = space_action == 1
        shift_held = shift_action == 1

        reward = 0
        
        # --- Handle Player Actions ---
        self._handle_input(movement, space_held, shift_held)
        
        # --- Update Game State ---
        self._update_targets()
        self._update_fired_lines()
        self._update_particles()
        
        # --- Spawn New Targets ---
        self._manage_target_spawning()
        
        # --- Collision Detection and Rewards ---
        hit_reward, miss_penalty = self._check_collisions()
        reward += hit_reward + miss_penalty

        # --- Aiming Reward ---
        reward += self._calculate_aiming_reward()
        
        self.steps += 1
        
        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.score >= self.SCORE_TO_WIN:
            reward += 100
            terminated = True
            self.game_over = True
            # sfx: game_win_sound
        elif self.misses >= self.MAX_MISSES:
            reward -= 100
            terminated = True
            self.game_over = True
            # sfx: game_over_sound
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Rotation
        rotation_speed = 2.0
        if movement == 3: # Left
            self.line_angle -= rotation_speed
        elif movement == 4: # Right
            self.line_angle += rotation_speed
        self.line_angle = max(190, min(350, self.line_angle)) # Clamp angle

        # Fire (on rising edge of space bar)
        if space_held and not self.previous_space_held:
            self._fire_line()
            # sfx: fire_laser_sound
        self.previous_space_held = space_held

        # Transform (on rising edge of shift key)
        if shift_held and not self.previous_shift_held:
            self.is_wide_mode = not self.is_wide_mode
            # sfx: transform_toggle_sound
        self.previous_shift_held = shift_held

    def _update_targets(self):
        # Increase difficulty over time
        if self.steps > 0 and self.steps % 100 == 0:
            self.base_target_speed += 0.05

        for target in self.targets:
            target.update()
        
        # Remove off-screen targets
        self.targets = [t for t in self.targets if -t.radius < t.pos.x < self.SCREEN_WIDTH + t.radius]

    def _update_fired_lines(self):
        for line in self.fired_lines:
            line.update(self.steps)

    def _update_particles(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifespan > 0 and p.radius > 0]

    def _check_collisions(self):
        hit_reward = 0
        miss_penalty = 0
        
        lines_to_remove = []
        targets_to_remove = []

        for line in self.fired_lines:
            if not line.active:
                continue

            # Check for hits
            for target in self.targets:
                if target in targets_to_remove:
                    continue
                
                dist = line.pos.distance_to(target.pos)
                hit_radius = target.radius + (5 if line.is_wide else 2)
                
                if dist < hit_radius:
                    self.score += 10
                    hit_reward += 10
                    targets_to_remove.append(target)
                    line.active = False
                    self._create_hit_particles(target.pos)
                    # sfx: target_hit_explosion
                    break # Line can only hit one target
            
            # Check for miss (off-screen)
            if line.active and not (0 < line.pos.y < self.SCREEN_HEIGHT and 0 < line.pos.x < self.SCREEN_WIDTH):
                self.misses += 1
                miss_penalty -= 5
                line.active = False
                # sfx: line_miss_whoosh

        # Clean up inactive lines that have faded
        self.fired_lines = [l for l in self.fired_lines if l.active or len(l.trail) > 1]
        self.targets = [t for t in self.targets if t not in targets_to_remove]
        
        return hit_reward, miss_penalty

    def _calculate_aiming_reward(self):
        if not self.targets:
            return 0
        
        player_base = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 10)
        
        # Find closest target
        closest_target = min(self.targets, key=lambda t: player_base.distance_to(t.pos))
        
        # Calculate angle to target
        angle_to_target_rad = math.atan2(closest_target.pos.y - player_base.y, closest_target.pos.x - player_base.x)
        angle_to_target_deg = math.degrees(angle_to_target_rad)
        
        # Normalize player angle to be in the same range as atan2
        player_angle_rad = math.radians(self.line_angle)
        
        # Calculate angular difference
        angle_diff = abs(angle_to_target_rad - player_angle_rad)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        # Give reward if aiming close enough
        if angle_diff < math.radians(5):
            return 0.1
        
        return 0

    def _fire_line(self):
        start_pos = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 10)
        line = FiredLine(start_pos, self.line_angle, self.is_wide_mode, self.steps)
        self.fired_lines.append(line)

    def _spawn_target(self):
        target = Target(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.base_target_speed)
        self.targets.append(target)

    def _manage_target_spawning(self):
        self.target_spawn_timer += 1
        # Spawn if not enough targets or timer runs out
        if len(self.targets) < 5 and self.target_spawn_timer > 60:
            self._spawn_target()
            self.target_spawn_timer = 0

    def _create_hit_particles(self, pos):
        for _ in range(30):
            angle = random.uniform(0, 360)
            speed = random.uniform(2, 6)
            vel = pygame.Vector2(math.cos(math.radians(angle)), math.sin(math.radians(angle))) * speed
            radius = random.uniform(3, 8)
            lifespan = random.randint(20, 40)
            self.particles.append(Particle(pos, vel, radius, self.COLOR_HIT, lifespan))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "misses": self.misses}

    def _render_game(self):
        self._render_grid()
        for target in self.targets:
            target.draw(self.screen)
        for line in self.fired_lines:
            line.draw(self.screen)
        self._render_player_line()
        for particle in self.particles:
            particle.draw(self.screen)
    
    def _render_grid(self):
        grid_spacing = 40
        oscillation = math.sin(self.steps * 0.05) * 25
        
        for i in range(self.SCREEN_WIDTH // grid_spacing + 2):
            x = i * grid_spacing + oscillation % grid_spacing - grid_spacing
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        
        for i in range(self.SCREEN_HEIGHT // grid_spacing + 1):
            y = i * grid_spacing
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_player_line(self):
        start_pos = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 10)
        length = 40
        end_pos_x = start_pos[0] + length * math.cos(math.radians(self.line_angle))
        end_pos_y = start_pos[1] + length * math.sin(math.radians(self.line_angle))
        
        width = 5 if self.is_wide_mode else 2
        pygame.draw.line(self.screen, self.COLOR_PLAYER, start_pos, (end_pos_x, end_pos_y), width)
        pygame.gfxdraw.filled_circle(self.screen, int(start_pos[0]), int(start_pos[1]), 8, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(start_pos[0]), int(start_pos[1]), 8, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_SCORE)
        self.screen.blit(score_text, (10, 10))
        
        # Misses
        miss_text = self.font_ui.render(f"MISSES: {self.misses}/{self.MAX_MISSES}", True, self.COLOR_UI_MISS)
        self.screen.blit(miss_text, (self.SCREEN_WIDTH - miss_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            if self.score >= self.SCORE_TO_WIN:
                end_text = "VICTORY!"
                color = (0, 255, 128)
            else:
                end_text = "GAME OVER"
                color = self.COLOR_UI_MISS
            
            text_surf = self.font_game_over.render(end_text, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == '__main__':
    # This block will not be executed in the testing environment, but is useful for local development.
    # To run, you will need to `pip install pygame`.
    # Un-comment the line below to run with a display.
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Transforming Line")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Game loop for human control
    while not done:
        # --- Human Input Mapping ---
        movement = 0 # none
        space_held = False
        shift_held = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = True
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = True
            
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Frame Rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'R' key
        
        clock.tick(30) # Run at 30 FPS

    env.close()