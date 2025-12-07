import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Particle:
    """A simple particle class for visual effects."""
    def __init__(self, x, y, vx, vy, radius, color, lifespan):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        # Optional: Add gravity or friction
        self.vx *= 0.98
        self.vy *= 0.98
        self.radius = max(0, self.radius * (self.lifespan / self.max_lifespan))

    def draw(self, surface, camera_x):
        if self.radius > 1:
            pos = (int(self.x - camera_x), int(self.y))
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(self.radius), self.color)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(self.radius), self.color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Hold [SPACE] to charge a jump, release to leap. Avoid red obstacles."
    )

    game_description = (
        "Guide a worm through a procedurally generated obstacle course to reach the exit portal. "
        "Time your jumps carefully to survive."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    
    # Colors
    COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
    COLOR_BG_BOTTOM = (0, 0, 139)    # Dark Blue
    COLOR_WORM = (50, 205, 50)       # Lime Green
    COLOR_WORM_OUTLINE = (34, 139, 34) # Forest Green
    COLOR_OBSTACLE = (255, 69, 0)    # OrangeRed
    COLOR_OBSTACLE_OUTLINE = (205, 55, 0)
    COLOR_EXIT = (75, 0, 130)        # Indigo
    COLOR_EXIT_PARTICLE = (148, 0, 211) # Dark Violet
    COLOR_TEXT = (255, 255, 255)
    COLOR_CHARGE_BAR = (255, 255, 0) # Yellow

    # Physics & Gameplay
    GRAVITY = 0.8
    WORM_BASE_SPEED = 4.0
    JUMP_CHARGE_RATE = 0.5
    MAX_JUMP_CHARGE = 20
    LEVEL_LENGTH = 10000
    GROUND_Y = SCREEN_HEIGHT - 40
    MAX_STEPS = 60 * FPS # 60 seconds

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
        self.font_gameover = pygame.font.SysFont("monospace", 48, bold=True)

        self.render_mode = render_mode
        self.game_over_message = ""

        # This will be initialized in reset()
        self.np_random = None
        self.worm_pos = None
        self.worm_vel = None
        self.worm_segments = None
        self.on_ground = None
        self.jump_charge = None
        self.was_space_held = None
        self.obstacles = None
        self.particles = None
        self.camera_x = None
        self.worm_speed = None
        self.steps = 0
        self.score = 0
        self.game_over = False


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""

        self.worm_pos = pygame.Vector2(150, self.GROUND_Y)
        self.worm_vel = pygame.Vector2(0, 0)
        # FIX: pygame.Vector2 does not have a .copy() method. Use the constructor to copy.
        self.worm_segments = [pygame.Vector2(self.worm_pos) - pygame.Vector2(i * 5, 0) for i in range(15)]
        self.worm_speed = self.WORM_BASE_SPEED
        
        self.on_ground = True
        self.jump_charge = 0
        self.was_space_held = False
        
        self.particles = []
        self.camera_x = 0

        self._generate_obstacles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        terminated = False
        
        if not self.game_over:
            # The action space doesn't use movement or shift, but we keep them for potential future use.
            # movement = action[0]
            space_held = action[1] == 1
            # shift_held = action[2] == 1

            self._handle_input(space_held)
            self._update_physics()
            
            # Difficulty scaling
            if self.steps > 0 and self.steps % 250 == 0:
                self.worm_speed += 0.1 # Slight speed increase

            # Check for events and calculate rewards
            reward += 0.01  # Small survival reward per step

            cleared_reward, _ = self._check_obstacle_clear()
            reward += cleared_reward

            collision = self._check_collisions()
            exit_reached = self._check_exit()
            timeout = self.steps >= self.MAX_STEPS

            if collision:
                reward = -10.0
                self.game_over = True
                terminated = True
                self.game_over_message = "CRASHED!"
                self._create_explosion(self.worm_pos, self.COLOR_OBSTACLE, 50)
            elif exit_reached:
                reward = 100.0
                self.game_over = True
                terminated = True
                self.game_over_message = "GOAL!"
            elif timeout:
                reward = -5.0
                self.game_over = True
                terminated = True
                self.game_over_message = "TIME OUT"

        self.score += reward
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # Truncated is always False in this environment
            self._get_info()
        )
    
    def _handle_input(self, space_held):
        # Charge jump
        if space_held and self.on_ground:
            self.jump_charge = min(self.MAX_JUMP_CHARGE, self.jump_charge + self.JUMP_CHARGE_RATE)

        # Release jump
        if not space_held and self.was_space_held and self.on_ground:
            self.worm_vel.y = -self.jump_charge
            self.on_ground = False
            self.jump_charge = 0
            self._create_explosion(pygame.Vector2(self.worm_pos.x, self.GROUND_Y), self.COLOR_WORM_OUTLINE, 10, -1)
        
        self.was_space_held = space_held

    def _update_physics(self):
        # Worm horizontal movement
        self.worm_pos.x += self.worm_speed
        
        # Worm vertical movement
        if not self.on_ground:
            self.worm_vel.y += self.GRAVITY
            self.worm_pos.y += self.worm_vel.y

        # Ground collision
        if self.worm_pos.y >= self.GROUND_Y:
            self.worm_pos.y = self.GROUND_Y
            self.worm_vel.y = 0
            if not self.on_ground:
                self._create_explosion(pygame.Vector2(self.worm_pos.x, self.GROUND_Y), self.COLOR_WORM_OUTLINE, 5, -0.5)
            self.on_ground = True

        # Update worm segments for undulating effect
        # FIX: pygame.Vector2 does not have a .copy() method. Use the constructor to copy.
        self.worm_segments.insert(0, pygame.Vector2(self.worm_pos))
        if len(self.worm_segments) > 15:
            self.worm_segments.pop()
        
        # This part of the original code caused erratic vertical movement of the entire worm body.
        # It's better to apply the sine wave offset during rendering for a visual-only effect.
        # for i, seg in enumerate(self.worm_segments):
        #     if i > 0:
        #         y_offset = math.sin(self.steps * 0.2 + i * 0.5) * 3
        #         seg.y += y_offset

        # Update camera to follow worm
        self.camera_x = self.worm_pos.x - 150

        # Update particles
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()
            
    def _generate_obstacles(self):
        self.obstacles = []
        current_x = 500
        while current_x < self.LEVEL_LENGTH - 500:
            gap_width = self.np_random.integers(150, 400)
            current_x += gap_width
            
            obstacle_width = self.np_random.integers(50, 150)
            obstacle_height = self.np_random.integers(50, 200)
            obstacle_rect = pygame.Rect(
                current_x, self.GROUND_Y - obstacle_height, obstacle_width, obstacle_height
            )
            
            # Store rect and a flag to track if reward was given
            self.obstacles.append({"rect": obstacle_rect, "rewarded": False})
            current_x += obstacle_width

    def _check_obstacle_clear(self):
        worm_head_rect = pygame.Rect(self.worm_pos.x - 5, self.worm_pos.y - 5, 10, 10)
        for obs in self.obstacles:
            if not obs["rewarded"] and worm_head_rect.left > obs["rect"].right:
                obs["rewarded"] = True
                return 1.0, True # Reward for clearing
        return 0.0, False

    def _check_collisions(self):
        worm_head_rect = pygame.Rect(self.worm_pos.x, self.worm_pos.y, 1, 1) # Check a single point for the head
        for obs in self.obstacles:
            if obs["rect"].colliderect(worm_head_rect):
                return True
        # Check world boundaries
        if self.worm_pos.y < 0: # Hit ceiling
            return True
        return False
        
    def _check_exit(self):
        return self.worm_pos.x > self.LEVEL_LENGTH

    def _create_explosion(self, pos, color, count, y_velocity_multiplier=1):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed * y_velocity_multiplier
            radius = self.np_random.uniform(2, 6)
            lifespan = self.np_random.integers(20, 40)
            self.particles.append(Particle(pos.x, pos.y, vx, vy, radius, color, lifespan))
    
    def _get_observation(self):
        self._render_background()
        self._render_world_elements()
        self._render_particles()
        self._render_worm()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_background(self):
        self.screen.fill(self.COLOR_BG_TOP)
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_WORM_OUTLINE, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

    def _render_world_elements(self):
        # Exit Portal
        exit_x = self.LEVEL_LENGTH - self.camera_x
        if exit_x < self.SCREEN_WIDTH + 50:
            portal_rect = pygame.Rect(exit_x, 0, 50, self.GROUND_Y)
            pygame.draw.rect(self.screen, self.COLOR_EXIT, portal_rect)
            # Portal particles
            if self.steps % 2 == 0 and self.np_random:
                p_x = exit_x + self.np_random.uniform(0, 50)
                p_y = self.np_random.uniform(0, self.GROUND_Y)
                p_vy = self.np_random.uniform(-2, 2)
                p_lifespan = self.np_random.integers(15, 30)
                p_radius = self.np_random.uniform(2, 5)
                self.particles.append(Particle(p_x + self.camera_x, p_y, 0, p_vy, p_radius, self.COLOR_EXIT_PARTICLE, p_lifespan))

        # Obstacles
        for obs in self.obstacles:
            rect = obs["rect"]
            screen_rect = rect.move(-self.camera_x, 0)
            if screen_rect.right > 0 and screen_rect.left < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, screen_rect, 3)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen, self.camera_x)

    def _render_worm(self):
        # Draw segments from tail to head
        for i in range(len(self.worm_segments) - 1, -1, -1):
            seg = self.worm_segments[i]
            radius = 8 - i * 0.2
            
            # Add a sine wave for visual undulation
            y_offset = math.sin(self.steps * 0.2 + i * 0.5) * 3 if self.on_ground else 0
            
            screen_pos = (int(seg.x - self.camera_x), int(seg.y - radius * 0.5 + y_offset))
            
            # Draw outline
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], int(radius) + 2, self.COLOR_WORM_OUTLINE)
            # Draw main body
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], int(radius), self.COLOR_WORM)

        # Draw eyes on the head
        head = self.worm_segments[0]
        head_screen_pos = (int(head.x - self.camera_x), int(head.y - 4))
        pygame.draw.circle(self.screen, (255, 255, 255), (head_screen_pos[0] + 2, head_screen_pos[1] - 2), 3)
        pygame.draw.circle(self.screen, (0, 0, 0), (head_screen_pos[0] + 3, head_screen_pos[1] - 2), 1)
        
        # Draw jump charge bar
        if self.jump_charge > 0:
            bar_width = 50
            bar_height = 8
            charge_ratio = self.jump_charge / self.MAX_JUMP_CHARGE
            
            bar_x = int(head.x - self.camera_x - bar_width / 2)
            bar_y = int(head.y - 25)
            
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_CHARGE_BAR, (bar_x, bar_y, bar_width * charge_ratio, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

    def _render_ui(self):
        # Time remaining
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 10))

        # Distance to exit
        dist = max(0, self.LEVEL_LENGTH - self.worm_pos.x) / 100 # In meters
        dist_text = self.font_ui.render(f"DIST: {dist:.0f}m", True, self.COLOR_TEXT)
        self.screen.blit(dist_text, (self.SCREEN_WIDTH - dist_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            game_over_text = self.font_gameover.render(self.game_over_message, True, self.COLOR_TEXT)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_to_exit": max(0, self.LEVEL_LENGTH - self.worm_pos.x),
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Player Controls ---
    # To play, run this file. Use SPACE to jump.
    
    obs, info = env.reset()
    terminated = False
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Worm Jumper")
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    total_reward = 0
    
    running = True
    while running:
        # Get human input
        keys = pygame.key.get_pressed()
        space_held = keys[pygame.K_SPACE]
        
        action = env.action_space.sample() # Start with a random action
        action[1] = 1 if space_held else 0 # Override with human input
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            real_screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated:
                print(f"Game Over! Final Score: {total_reward:.2f}")

    env.close()