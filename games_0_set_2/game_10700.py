import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:48:26.366646
# Source Brief: brief_00700.md
# Brief Index: 700
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
        "Guide a charged particle through a series of complex mazes using electromagnetic coils. "
        "Reach the exit before time runs out or the particle becomes too unstable."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to activate the corresponding electromagnetic coil "
        "and guide the particle through the maze."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    LEVEL_TIME_LIMIT = 30  # seconds

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_WALL = (40, 70, 120)
    COLOR_WALL_BORDER = (80, 110, 160)
    COLOR_EXIT = (0, 255, 128)
    COLOR_EXIT_GLOW = (150, 255, 200)
    COLOR_PARTICLE = (255, 255, 0)
    COLOR_PARTICLE_GLOW = (255, 200, 0)
    COLOR_COIL_INACTIVE = (60, 60, 70)
    COLOR_COIL_ACTIVE = (220, 220, 255)
    COLOR_FIELD_LINES = (150, 180, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_TIMER_WARN = (255, 100, 100)

    # Physics
    FORCE_MAGNITUDE = 0.04
    PARTICLE_BASE_RADIUS = 8
    PARTICLE_DAMPING = 0.995 # slight velocity decay
    WALL_BOUNCE_DAMPING = 0.75
    MAX_CHARGE = 5.0

    # Rewards
    REWARD_GOAL = 100.0
    REWARD_TIMEOUT = -100.0
    REWARD_INSTABILITY = -50.0
    REWARD_WALL_HIT = -1.0
    REWARD_DISTANCE_MULTIPLIER = 0.2

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_level = pygame.font.SysFont("monospace", 24, bold=True)

        self.mazes, self.starts, self.exits = self._generate_mazes()

        # State variables initialized in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.timer = 0
        self.particle_pos = pygame.math.Vector2(0, 0)
        self.particle_vel = pygame.math.Vector2(0, 0)
        self.particle_charge = 0
        self.active_coils = [False] * 4  # up, down, left, right
        self.exit_rect = pygame.Rect(0, 0, 0, 0)
        self.coil_hitboxes = []
        self.wall_hit_effects = []
        
        # self.reset() is called by the environment wrapper

    def _generate_mazes(self):
        w, h = self.SCREEN_WIDTH, self.SCREEN_HEIGHT
        pad = 50
        wall_thickness = 10
        
        # Maze definitions (list of pygame.Rect)
        mazes = [
            # Level 1: Simple corridor
            [pygame.Rect(pad, pad, w - 2 * pad, wall_thickness),
             pygame.Rect(pad, h - pad - wall_thickness, w - 2 * pad, wall_thickness)],
            # Level 2: One turn
            [pygame.Rect(pad, pad, w - 2 * pad, wall_thickness),
             pygame.Rect(pad, h - pad - wall_thickness, w / 2, wall_thickness),
             pygame.Rect(w / 2, pad + wall_thickness, wall_thickness, h - 2 * pad - wall_thickness)],
            # Level 3: S-curve
            [pygame.Rect(pad, pad, w - 2 * pad, wall_thickness),
             pygame.Rect(pad, h - pad - wall_thickness, w - 2 * pad, wall_thickness),
             pygame.Rect(w / 2 - 50, pad, wall_thickness, h / 2 - pad + 50),
             pygame.Rect(w / 2 + 50, h / 2, wall_thickness, h / 2 - pad)],
            # Level 4: Central block
            [pygame.Rect(pad, pad, w - 2 * pad, wall_thickness),
             pygame.Rect(pad, h - pad - wall_thickness, w - 2 * pad, wall_thickness),
             pygame.Rect(pad, pad, wall_thickness, h - 2 * pad),
             pygame.Rect(w - pad - wall_thickness, pad, wall_thickness, h - 2 * pad),
             pygame.Rect(w/2 - 100, h/2 - 50, 200, 100)],
            # Level 5: Spiral
            [pygame.Rect(pad, pad, w - 2 * pad, wall_thickness),
             pygame.Rect(pad, h - pad - wall_thickness, w - 2 * pad, wall_thickness),
             pygame.Rect(pad, pad, wall_thickness, h - 2 * pad),
             pygame.Rect(w - pad - wall_thickness, pad, wall_thickness, h - pad * 3),
             pygame.Rect(pad + 100, pad + 100, w - 2 * (pad + 100), wall_thickness),
             pygame.Rect(pad + 100, h - (pad + 100) - wall_thickness, w - 2 * (pad + 100), wall_thickness),
             pygame.Rect(pad + 100, pad + 100, wall_thickness, h - 2 * (pad + 100))]
        ]
        
        starts = [
            (pad + 30, h / 2), (pad + 30, h - pad - 30), (pad + 30, pad + 30),
            (pad + 30, h / 2), (pad + 30, h / 2)
        ]
        exits = [
            (w - pad - 30, h / 2), (w / 2 + 30, pad + 30), (w - pad - 30, h - pad - 30),
            (w - pad - 30, h / 2), (w/2, h/2)
        ]
        return mazes, starts, exits

    def _reset_level(self):
        self.timer = self.LEVEL_TIME_LIMIT
        self.particle_charge = 1.0
        self.wall_hit_effects = []
        
        level_idx = self.level - 1
        self.particle_pos = pygame.math.Vector2(self.starts[level_idx])
        
        initial_speed = 0.05 * level_idx
        random_angle = self.np_random.uniform(0, 2 * math.pi)
        self.particle_vel = pygame.math.Vector2(
            math.cos(random_angle) * initial_speed,
            math.sin(random_angle) * initial_speed
        )

        self.exit_pos = pygame.math.Vector2(self.exits[level_idx])
        self.exit_rect = pygame.Rect(self.exit_pos.x - 20, self.exit_pos.y - 20, 40, 40)

        # Define coil positions
        coil_radius = 15
        self.coil_positions = [
            (self.SCREEN_WIDTH / 2, coil_radius),  # Up
            (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - coil_radius),  # Down
            (coil_radius, self.SCREEN_HEIGHT / 2),  # Left
            (self.SCREEN_WIDTH - coil_radius, self.SCREEN_HEIGHT / 2)  # Right
        ]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        
        self._reset_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0
        
        # 1. Unpack action and update coil states
        movement = action[0]
        self.active_coils = [movement == i + 1 for i in range(4)]

        # 2. Store pre-physics state for reward calculation
        prev_dist_to_exit = self.particle_pos.distance_to(self.exit_pos)

        # 3. Update physics
        self._handle_physics()

        # 4. Handle collisions
        wall_hit = self._handle_collisions()
        if wall_hit:
            reward += self.REWARD_WALL_HIT
            self.particle_charge = min(self.MAX_CHARGE, self.particle_charge + 0.5)
            self.timer = max(0, self.timer - 0.5)

        # 5. Check for goal
        reached_exit = self.exit_rect.collidepoint(self.particle_pos.x, self.particle_pos.y)

        # 6. Update timer
        self.timer -= 1.0 / self.FPS

        # 7. Calculate reward
        new_dist_to_exit = self.particle_pos.distance_to(self.exit_pos)
        reward += (prev_dist_to_exit - new_dist_to_exit) * self.REWARD_DISTANCE_MULTIPLIER
        
        terminated = False
        if reached_exit:
            time_bonus = max(0, self.timer / self.LEVEL_TIME_LIMIT)
            level_reward = self.REWARD_GOAL * (0.5 + 0.5 * time_bonus)
            reward += level_reward
            self.score += level_reward
            
            if self.level >= len(self.mazes):
                self.game_over = True
                terminated = True
            else:
                self.level += 1
                self._reset_level()
        
        # 8. Check termination conditions
        if not terminated:
            terminated = self._check_termination()
            if terminated:
                if self.timer <= 0:
                    reward = self.REWARD_TIMEOUT
                elif self.particle_charge >= self.MAX_CHARGE:
                    reward = self.REWARD_INSTABILITY
                self.game_over = True

        self.score += reward # Add frame-by-frame rewards to total score

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_physics(self):
        force = pygame.math.Vector2(0, 0)
        directions = [
            pygame.math.Vector2(0, -1), # Up
            pygame.math.Vector2(0, 1),  # Down
            pygame.math.Vector2(-1, 0), # Left
            pygame.math.Vector2(1, 0)   # Right
        ]
        
        for i, is_active in enumerate(self.active_coils):
            if is_active:
                force += directions[i]
        
        if force.length_squared() > 0:
            force.normalize_ip()
            force *= self.FORCE_MAGNITUDE
        
        self.particle_vel += force
        self.particle_vel *= self.PARTICLE_DAMPING
        self.particle_pos += self.particle_vel

    def _handle_collisions(self):
        hit = False
        particle_rect = pygame.Rect(
            self.particle_pos.x - self.PARTICLE_BASE_RADIUS,
            self.particle_pos.y - self.PARTICLE_BASE_RADIUS,
            self.PARTICLE_BASE_RADIUS * 2,
            self.PARTICLE_BASE_RADIUS * 2
        )

        for wall in self.mazes[self.level - 1]:
            if particle_rect.colliderect(wall):
                hit = True
                # Find overlap to push particle out
                delta_x = self.particle_pos.x - wall.centerx
                delta_y = self.particle_pos.y - wall.centery
                
                # Determine collision edge
                if abs(delta_x) / wall.width > abs(delta_y) / wall.height: # Horizontal collision
                    if self.particle_vel.x * delta_x > 0: # Moving into the wall
                        self.particle_vel.x *= -self.WALL_BOUNCE_DAMPING
                    # Push out
                    self.particle_pos.x = wall.right + self.PARTICLE_BASE_RADIUS if delta_x > 0 else wall.left - self.PARTICLE_BASE_RADIUS
                else: # Vertical collision
                    if self.particle_vel.y * delta_y > 0: # Moving into the wall
                        self.particle_vel.y *= -self.WALL_BOUNCE_DAMPING
                    # Push out
                    self.particle_pos.y = wall.bottom + self.PARTICLE_BASE_RADIUS if delta_y > 0 else wall.top - self.PARTICLE_BASE_RADIUS
                
                self.wall_hit_effects.append({'pos': self.particle_pos.copy(), 'life': 10})

        # Boundary checks
        if self.particle_pos.x < self.PARTICLE_BASE_RADIUS or self.particle_pos.x > self.SCREEN_WIDTH - self.PARTICLE_BASE_RADIUS:
            self.particle_vel.x *= -1
            self.particle_pos.x = np.clip(self.particle_pos.x, self.PARTICLE_BASE_RADIUS, self.SCREEN_WIDTH - self.PARTICLE_BASE_RADIUS)
        if self.particle_pos.y < self.PARTICLE_BASE_RADIUS or self.particle_pos.y > self.SCREEN_HEIGHT - self.PARTICLE_BASE_RADIUS:
            self.particle_vel.y *= -1
            self.particle_pos.y = np.clip(self.particle_pos.y, self.PARTICLE_BASE_RADIUS, self.SCREEN_HEIGHT - self.PARTICLE_BASE_RADIUS)
            
        return hit

    def _check_termination(self):
        if self.timer <= 0:
            return True
        if self.particle_charge >= self.MAX_CHARGE:
            return True
        if self.level > len(self.mazes):
            return True
        # A simple step limit to prevent infinite loops
        if self.steps >= 5000:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_coils_and_fields()
        self._render_maze()
        self._render_exit()
        self._render_particle()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "timer": self.timer,
            "charge": self.particle_charge,
        }

    def _render_maze(self):
        for wall in self.mazes[self.level - 1]:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
            pygame.draw.rect(self.screen, self.COLOR_WALL_BORDER, wall, 1)

    def _render_coils_and_fields(self):
        for i, pos in enumerate(self.coil_positions):
            color = self.COLOR_COIL_ACTIVE if self.active_coils[i] else self.COLOR_COIL_INACTIVE
            pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), 10)
            
            if self.active_coils[i]:
                # Pulsating field lines
                pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
                for j in range(3):
                    radius = 20 + j * 15 + pulse * 5
                    alpha = int(max(0, 100 - radius * 1.5 + pulse * 20))
                    if alpha > 0:
                        try:
                            pygame.gfxdraw.arc(self.screen, int(pos[0]), int(pos[1]), int(radius), 0, 360, (*self.COLOR_FIELD_LINES, alpha))
                        except TypeError: # Sometimes alpha can be invalid
                            pass


    def _render_exit(self):
        # Glow effect
        glow_size = int(self.exit_rect.width * (1.2 + 0.1 * math.sin(self.steps * 0.1)))
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_EXIT_GLOW, 50), (glow_size // 2, glow_size // 2), glow_size // 2)
        self.screen.blit(glow_surf, (self.exit_rect.centerx - glow_size // 2, self.exit_rect.centery - glow_size // 2))
        
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect, border_radius=5)

    def _render_particle(self):
        pos = (int(self.particle_pos.x), int(self.particle_pos.y))
        radius = int(self.PARTICLE_BASE_RADIUS * (1 + (self.particle_charge - 1) * 0.2))
        
        # Glow effect
        glow_radius = int(radius * (2.0 + (self.particle_charge / self.MAX_CHARGE)))
        glow_alpha = int(50 + 100 * (self.particle_charge / self.MAX_CHARGE))
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PARTICLE_GLOW, glow_alpha), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

        # Core particle
        pygame.draw.circle(self.screen, self.COLOR_PARTICLE, pos, radius)

    def _render_effects(self):
        for effect in self.wall_hit_effects[:]:
            effect['life'] -= 1
            if effect['life'] <= 0:
                self.wall_hit_effects.remove(effect)
                continue
            
            alpha = int(255 * (effect['life'] / 10))
            size = int(10 * (1 - (effect['life'] / 10)))
            pygame.draw.circle(self.screen, (*self.COLOR_PARTICLE, alpha), (int(effect['pos'].x), int(effect['pos'].y)), size)


    def _render_ui(self):
        # Level
        level_text = self.font_level.render(f"LEVEL: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (10, 10))
        
        # Timer
        timer_color = self.COLOR_TIMER_WARN if self.timer < 5 else self.COLOR_UI_TEXT
        timer_text = self.font_ui.render(f"TIMER: {max(0, self.timer):.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))
        
        # Charge
        charge_text = self.font_ui.render(f"CHARGE: {self.particle_charge/self.MAX_CHARGE:.0%}", True, self.COLOR_UI_TEXT)
        self.screen.blit(charge_text, (self.SCREEN_WIDTH - charge_text.get_width() - 10, 35))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score:.0f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, self.SCREEN_HEIGHT - score_text.get_height() - 10))

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        print("Running implementation validation...")
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

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Make sure to unset the dummy video driver if you want to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Magnetic Maze")
    clock = pygame.time.Clock()
    
    running = True
    total_score = 0
    
    while running:
        action = [0, 0, 0] # Default action: no movement, no buttons
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if keys[pygame.K_r]: # Press R to reset
            obs, info = env.reset()
            total_score = 0
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_score += reward
        
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.0f}, Level: {info['level']}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_score = 0

        clock.tick(env.FPS)
        
    env.close()