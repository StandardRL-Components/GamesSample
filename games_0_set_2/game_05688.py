# Generated: 2025-08-28T05:46:28.788856
# Source Brief: brief_05688.md
# Brief Index: 5688

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Ensure Pygame runs in a headless mode for the environment
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: ←→ to run. ↑ or Space to jump (Space is higher). Avoid red obstacles!"
    )

    # Short, user-facing description of the game
    game_description = (
        "A fast-paced side-scrolling platformer. Guide your robot to the finish line, "
        "avoiding projectiles and pits. Faster times and less damage give higher scores."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 50)
    COLOR_ROBOT = (50, 150, 255)
    COLOR_ROBOT_GLOW = (50, 150, 255)
    COLOR_PLATFORM = (85, 85, 105)
    COLOR_PROJECTILE = (255, 50, 50)
    COLOR_PROJECTILE_GLOW = (255, 50, 50)
    COLOR_FINISH_LINE = (50, 255, 50)
    COLOR_TEXT = (220, 220, 220)

    # Physics & Game Parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WORLD_WIDTH = 6000
    FPS = 30
    GRAVITY = 0.8
    ROBOT_MOVE_SPEED = 6
    ROBOT_JUMP_STRENGTH = -14
    ROBOT_HIGH_JUMP_STRENGTH = -18
    MAX_FALL_SPEED = 15
    MAX_DAMAGE = 5
    MAX_STEPS = 5000
    INVULNERABILITY_FRAMES = 60 # 2 seconds
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Initialize state variables
        self.robot_pos = None
        self.robot_vel = None
        self.robot_on_ground = None
        self.robot_damage = None
        self.robot_invulnerable_timer = None
        self.robot_last_x = None

        self.platforms = []
        self.projectiles = []
        self.particles = []
        
        self.camera_x = 0
        self.finish_line_x = self.WORLD_WIDTH - 200

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # self.reset() is called by the wrapper or test harness, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.robot_pos = pygame.Vector2(200, 200)
        self.robot_vel = pygame.Vector2(0, 0)
        self.robot_on_ground = False
        self.robot_damage = 0
        self.robot_invulnerable_timer = 0
        self.robot_last_x = self.robot_pos.x
        
        self.camera_x = 0
        self._generate_level()
        
        self.projectiles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        # Unpack action
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Game Logic ---
        reward = 0
        terminated = False

        if not self.game_over:
            # 1. Handle Input
            self._handle_input(movement, space_held)
            
            # 2. Update Physics & State
            self._update_robot_physics()
            self._update_projectiles()
            self._update_particles()
            self._spawn_projectiles()
            
            # 3. Update Camera
            target_camera_x = self.robot_pos.x - self.SCREEN_WIDTH / 3
            self.camera_x += (target_camera_x - self.camera_x) * 0.1
            self.camera_x = max(0, min(self.camera_x, self.WORLD_WIDTH - self.SCREEN_WIDTH))

            # 4. Calculate Reward
            reward += self._calculate_reward()
            self.robot_last_x = self.robot_pos.x

            # 5. Check Termination Conditions
            terminated, term_reward = self._check_termination()
            reward += term_reward
            if terminated:
                self.game_over = True

        self.steps += 1
        self.score += reward
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Per Gymnasium API, if truncated, terminated should also be true
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Horizontal movement
        if movement == 3:  # Left
            self.robot_vel.x = -self.ROBOT_MOVE_SPEED
        elif movement == 4:  # Right
            self.robot_vel.x = self.ROBOT_MOVE_SPEED
        else:
            # Apply friction
            self.robot_vel.x *= 0.85
            if abs(self.robot_vel.x) < 0.1:
                self.robot_vel.x = 0

        # Jumping
        if self.robot_on_ground:
            if space_held:
                self.robot_vel.y = self.ROBOT_HIGH_JUMP_STRENGTH
                self.robot_on_ground = False
                self._create_particles(self.robot_pos + pygame.Vector2(0, 20), 10, self.COLOR_PLATFORM) # Jump dust
            elif movement == 1: # Up
                self.robot_vel.y = self.ROBOT_JUMP_STRENGTH
                self.robot_on_ground = False
                self._create_particles(self.robot_pos + pygame.Vector2(0, 20), 5, self.COLOR_PLATFORM) # Jump dust

    def _update_robot_physics(self):
        # Apply gravity
        self.robot_vel.y += self.GRAVITY
        self.robot_vel.y = min(self.robot_vel.y, self.MAX_FALL_SPEED)

        # Move horizontally
        self.robot_pos.x += self.robot_vel.x
        self.robot_pos.x = max(15, min(self.robot_pos.x, self.WORLD_WIDTH - 15))
        
        robot_rect = self._get_robot_rect()
        for plat in self.platforms:
            if robot_rect.colliderect(plat):
                if self.robot_vel.x > 0: # Moving right
                    robot_rect.right = plat.left
                    self.robot_pos.x = robot_rect.centerx
                    self.robot_vel.x = 0
                elif self.robot_vel.x < 0: # Moving left
                    robot_rect.left = plat.right
                    self.robot_pos.x = robot_rect.centerx
                    self.robot_vel.x = 0
        
        # Move vertically
        self.robot_pos.y += self.robot_vel.y
        robot_rect = self._get_robot_rect()
        self.robot_on_ground = False
        for plat in self.platforms:
            if robot_rect.colliderect(plat):
                if self.robot_vel.y > 0: # Moving down
                    robot_rect.bottom = plat.top
                    self.robot_pos.y = robot_rect.centery
                    self.robot_vel.y = 0
                    self.robot_on_ground = True
                elif self.robot_vel.y < 0: # Moving up
                    robot_rect.top = plat.bottom
                    self.robot_pos.y = robot_rect.centery
                    self.robot_vel.y = 0

        # Update invulnerability timer
        if self.robot_invulnerable_timer > 0:
            self.robot_invulnerable_timer -= 1

        # Check projectile collisions
        if self.robot_invulnerable_timer == 0:
            robot_rect = self._get_robot_rect() # Re-get rect after position updates
            for proj in self.projectiles[:]:
                if robot_rect.colliderect(proj['rect']):
                    self.projectiles.remove(proj)
                    self.robot_damage += 1
                    self.robot_invulnerable_timer = self.INVULNERABILITY_FRAMES
                    self._create_particles(self.robot_pos, 20, self.COLOR_PROJECTILE, 1.5)
                    break
    
    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj['pos'].x += proj['vel'].x
            proj['rect'].center = proj['pos']
            # Check if projectile is off-screen relative to the camera
            world_rect = proj['rect'].move(-self.camera_x, 0)
            if not self.screen.get_rect().colliderect(world_rect):
                self.projectiles.remove(proj)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _spawn_projectiles(self):
        # Difficulty scaling: projectile speed increases over time
        projectile_speed = 5 + (self.steps / 500) * 0.5
        
        if self.np_random.random() < 0.02 and self.steps > 100:
            spawn_x = self.camera_x + self.SCREEN_WIDTH + 50
            spawn_y = self.np_random.integers(50, self.SCREEN_HEIGHT - 50)
            
            # Ensure projectile doesn't spawn inside a platform
            spawn_rect = pygame.Rect(spawn_x - 5, spawn_y - 5, 10, 10)
            can_spawn = True
            for plat in self.platforms:
                if plat.colliderect(spawn_rect):
                    can_spawn = False
                    break
            
            if can_spawn:
                new_proj = {
                    'pos': pygame.Vector2(spawn_x, spawn_y),
                    'vel': pygame.Vector2(-projectile_speed, 0),
                    'rect': pygame.Rect(0, 0, 10, 10)
                }
                new_proj['rect'].center = new_proj['pos']
                self.projectiles.append(new_proj)

    def _calculate_reward(self):
        reward = 0
        # Reward for moving towards the finish line
        progress_made = self.robot_pos.x - self.robot_last_x
        if progress_made > 0:
            reward += 0.01 * progress_made
        
        return reward

    def _check_termination(self):
        # Win condition: reach finish line
        if self.robot_pos.x >= self.finish_line_x:
            reward = 5
            if self.robot_damage == 0:
                reward += 100 # Flawless victory bonus
            return True, reward

        # Lose condition: fall into a pit
        if self.robot_pos.y > self.SCREEN_HEIGHT + 50:
            return True, -10

        # Lose condition: too much damage
        if self.robot_damage >= self.MAX_DAMAGE:
            return True, -10

        # Damage penalty (event-based)
        if self.robot_invulnerable_timer == self.INVULNERABILITY_FRAMES - 1:
             return False, -1.0 # Just took damage

        return False, 0

    def _generate_level(self):
        self.platforms = []
        # Starting platform
        self.platforms.append(pygame.Rect(0, 350, 400, 50))
        
        current_x = 400
        current_y = 350
        
        while current_x < self.WORLD_WIDTH - 300:
            gap = self.np_random.integers(80, 150)
            width = self.np_random.integers(150, 400)
            
            # Ensure height change is jumpable
            y_change = self.np_random.integers(-120, 120)
            next_y = np.clip(current_y + y_change, 150, self.SCREEN_HEIGHT - 50)
            
            current_x += gap
            self.platforms.append(pygame.Rect(current_x, next_y, width, self.SCREEN_HEIGHT - next_y))
            current_x += width
            current_y = next_y

        # Final platform
        self.platforms.append(pygame.Rect(self.finish_line_x - 100, 350, 300, 50))

    def _get_robot_rect(self):
        return pygame.Rect(self.robot_pos.x - 15, self.robot_pos.y - 20, 30, 40)
        
    def _get_observation(self):
        # --- Rendering ---
        # 1. Background
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        
        # 2. Game Elements
        self._render_game()
        
        # 3. UI
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw grid
        grid_spacing = 50
        offset_x = -self.camera_x % grid_spacing
        for x in range(int(offset_x), self.SCREEN_WIDTH, grid_spacing):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, grid_spacing):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Render particles
        for p in self.particles:
            pos = (int(p['pos'].x - self.camera_x), int(p['pos'].y))
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            try:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), color)
            except TypeError: # Sometimes color alpha can be invalid
                pass


        # Render platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat.move(-self.camera_x, 0))

        # Render finish line
        finish_rect = pygame.Rect(self.finish_line_x - self.camera_x, 0, 10, self.SCREEN_HEIGHT)
        for i in range(0, self.SCREEN_HEIGHT, 20):
             pygame.draw.rect(self.screen, self.COLOR_FINISH_LINE, (finish_rect.x, i, 10, 10))
        
        # Render projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'].x - self.camera_x), int(proj['pos'].y))
            # Glow effect
            for i in range(4, 0, -1):
                alpha = 80 - i * 20
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5 + i*2, (*self.COLOR_PROJECTILE_GLOW, alpha))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, self.COLOR_PROJECTILE)

        # Render robot
        robot_rect_cam = self._get_robot_rect().move(-self.camera_x, 0)
        
        # Invulnerability flash
        if self.robot_invulnerable_timer > 0 and (self.robot_invulnerable_timer // 3) % 2 == 0:
            return # Don't draw robot to make it flash

        # Bobbing animation
        bob = 0
        if self.robot_on_ground and abs(self.robot_vel.x) > 0.1:
            bob = math.sin(self.steps * 0.5) * 2
        
        # Jump stretch
        stretch = 0
        if not self.robot_on_ground:
            stretch = -min(5, self.robot_vel.y * 0.2)

        robot_draw_rect = pygame.Rect(
            robot_rect_cam.x,
            robot_rect_cam.y - bob + stretch,
            robot_rect_cam.width,
            robot_rect_cam.height - stretch * 2
        )

        # Glow effect
        for i in range(8, 0, -1):
            alpha = 60 - i * 7
            pygame.gfxdraw.box(self.screen, robot_draw_rect.inflate(i*2, i*2), (*self.COLOR_ROBOT_GLOW, alpha))

        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_draw_rect, border_radius=5)
        
        # Damage cracks
        if self.robot_damage > 0:
            self._draw_crack(robot_draw_rect, (0.7, 0.2), (0.9, 0.4), 1)
        if self.robot_damage > 1:
            self._draw_crack(robot_draw_rect, (0.2, 0.3), (0.4, 0.1), 2)
        if self.robot_damage > 2:
            self._draw_crack(robot_draw_rect, (0.3, 0.8), (0.1, 0.6), 3)
        if self.robot_damage > 3:
            self._draw_crack(robot_draw_rect, (0.8, 0.7), (0.6, 0.9), 4)

    def _draw_crack(self, rect, start_rel, end_rel, seed):
        rng = random.Random(seed)
        start_pos = (rect.x + rect.width * start_rel[0], rect.y + rect.height * start_rel[1])
        end_pos = (rect.x + rect.width * end_rel[0], rect.y + rect.height * end_rel[1])
        mid_pos = (
            (start_pos[0] + end_pos[0]) / 2 + rng.uniform(-5, 5),
            (start_pos[1] + end_pos[1]) / 2 + rng.uniform(-5, 5)
        )
        pygame.draw.aaline(self.screen, self.COLOR_BG, start_pos, mid_pos, 2)
        pygame.draw.aaline(self.screen, self.COLOR_BG, mid_pos, end_pos, 2)

    def _render_ui(self):
        # Timer
        time_elapsed = self.steps / self.FPS
        timer_text = f"TIME: {time_elapsed:.2f}"
        timer_surf = self.font_main.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 15, 10))

        # Damage
        damage_text = f"DAMAGE: {self.robot_damage}/{self.MAX_DAMAGE}"
        damage_surf = self.font_main.render(damage_text, True, self.COLOR_TEXT)
        self.screen.blit(damage_surf, (15, 10))

        # Progress bar
        progress = self.robot_pos.x / self.finish_line_x
        bar_width = self.SCREEN_WIDTH - 40
        bar_height = 8
        pygame.draw.rect(self.screen, self.COLOR_GRID, (20, self.SCREEN_HEIGHT - 20, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, (20, self.SCREEN_HEIGHT - 20, bar_width * progress, bar_height), border_radius=4)
        
        # Finish line marker on progress bar
        finish_marker_x = 20 + bar_width
        pygame.draw.rect(self.screen, self.COLOR_FINISH_LINE, (finish_marker_x, self.SCREEN_HEIGHT - 24, 4, bar_height + 8), border_radius=2)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "damage": self.robot_damage,
            "progress": self.robot_pos.x / self.finish_line_x if self.finish_line_x > 0 else 1.0
        }
        
    def _create_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 * speed_mult
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pygame.Vector2(pos), # FIX: Create a new vector, don't use a reference
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color,
                'size': self.np_random.random() * 3 + 1
            })

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # --- Example of how to run the environment ---
    
    # Un-set the headless environment variable to allow display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Use a window to display the game
    pygame.display.set_caption("Robot Platformer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    # Map keyboard keys to MultiDiscrete actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while running:
        # --- Human Controls ---
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0 # Unused in this game
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]:
            movement_action = key_map[pygame.K_LEFT]
        elif keys[pygame.K_RIGHT]:
            movement_action = key_map[pygame.K_RIGHT]
        
        if keys[pygame.K_UP]:
            movement_action = key_map[pygame.K_UP]
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        action = [movement_action, space_action, shift_action]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}. Info: {info}")
            obs, info = env.reset()
            total_reward = 0
            # Add a small delay on reset to make it noticeable
            pygame.time.wait(500)
            
    env.close()