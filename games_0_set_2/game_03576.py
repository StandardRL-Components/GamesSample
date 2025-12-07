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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move in the air. Space for a large jump, Shift for a small jump. ↓ to fall faster."
    )

    game_description = (
        "Hop between procedurally generated platforms to reach the top. Platforms move and the layout changes over time. Don't fall or run out of time!"
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 50
    MAX_TIME_SECONDS = 180
    MAX_STEPS = MAX_TIME_SECONDS * FPS

    # Colors
    COLOR_BG_TOP = (40, 50, 90)
    COLOR_BG_BOTTOM = (10, 10, 30)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLATFORM = (120, 120, 140)
    COLOR_GOAL_PLATFORM = (255, 215, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_PARTICLE = (200, 200, 220)

    # Player Physics
    GRAVITY = 0.25
    PLAYER_SIZE = 16
    LARGE_JUMP_POWER = -8.0
    SMALL_JUMP_POWER = -6.0
    AIR_CONTROL_FORCE = 0.35
    FAST_FALL_FORCE = 0.4
    MAX_VEL_X = 4.0
    MAX_VEL_Y = 8.0

    # Platform Generation
    PLATFORM_MIN_WIDTH, PLATFORM_MAX_WIDTH = 60, 120
    PLATFORM_HEIGHT = 12
    PLATFORM_COUNT = 15
    STAGE_DURATION_SECONDS = 60
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_large = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 16)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 20)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.platforms = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_remaining = None
        self.max_height_reached = None
        self.current_stage = None
        self.platform_base_speed = None
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_TIME_SECONDS
        self.max_height_reached = self.HEIGHT
        self.current_stage = 0
        self.platform_base_speed = 0.5

        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.on_ground = True
        
        self.particles = []
        self._generate_platforms()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            # After game is over, just return the final state
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        reward = self._update_game_state(action)
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.game_over: # Assign terminal rewards only once
            if self.player_pos.y >= self.HEIGHT: # Fell off
                reward -= 100
                self.score -= 100
            elif self.time_remaining <= 0: # Timed out
                reward -= 10
                self.score -= 10
            # Win reward is handled in collision detection
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        step_reward = 0.1  # Survival reward

        # --- Time and Stage Management ---
        self.time_remaining -= 1 / self.FPS
        
        new_stage = int(self.steps / (self.STAGE_DURATION_SECONDS * self.FPS))
        if new_stage > self.current_stage:
            self.current_stage = new_stage
            self.platform_base_speed = min(2.0, 0.5 + new_stage * 0.05)
            self._generate_platforms()

        # --- Player Update ---
        # Handle Jumps
        if self.on_ground:
            if space_held: # Large jump
                self.player_vel.y = self.LARGE_JUMP_POWER
                self.on_ground = False
            elif shift_held: # Small jump
                self.player_vel.y = self.SMALL_JUMP_POWER
                self.on_ground = False

        # Apply Gravity
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY

        # Air Control and Fast Fall
        if not self.on_ground:
            if movement == 3: # Left
                self.player_vel.x -= self.AIR_CONTROL_FORCE
            elif movement == 4: # Right
                self.player_vel.x += self.AIR_CONTROL_FORCE
            elif movement == 2: # Down
                self.player_vel.y += self.FAST_FALL_FORCE
        
        # Friction
        self.player_vel.x *= 0.95 

        # Clamp Velocity
        self.player_vel.x = max(-self.MAX_VEL_X, min(self.MAX_VEL_X, self.player_vel.x))
        self.player_vel.y = max(-self.MAX_VEL_Y, min(self.MAX_VEL_Y, self.player_vel.y))

        # Update Position
        self.player_pos += self.player_vel
        
        # Screen Bounds
        if self.player_pos.x < 0:
            self.player_pos.x = 0
            self.player_vel.x = 0
        if self.player_pos.x > self.WIDTH - self.PLAYER_SIZE:
            self.player_pos.x = self.WIDTH - self.PLAYER_SIZE
            self.player_vel.x = 0

        # --- Platform & Collision Update ---
        self._update_platforms()
        collision_reward = self._handle_collisions()
        step_reward += collision_reward

        # --- Height Reward ---
        if self.player_pos.y < self.max_height_reached:
            step_reward += 5
            self.score += 5
            self.max_height_reached = self.player_pos.y

        # --- Particle Update ---
        self._update_particles()
        
        self.score += step_reward
        return step_reward

    def _generate_platforms(self):
        self.platforms = []
        
        # Start platform
        start_plat_width = 120
        start_plat = {
            "rect": pygame.Rect(self.WIDTH / 2 - start_plat_width / 2, self.HEIGHT - 40, start_plat_width, self.PLATFORM_HEIGHT),
            "speed": 0, "dir": 1, "is_goal": False
        }
        self.platforms.append(start_plat)
        
        last_y = start_plat["rect"].y
        
        for i in range(self.PLATFORM_COUNT):
            y_spacing = self.np_random.integers(40, 80)
            y = last_y - y_spacing
            
            x = self.np_random.integers(0, self.WIDTH - self.PLATFORM_MIN_WIDTH)
            
            width = self.np_random.integers(self.PLATFORM_MIN_WIDTH, self.PLATFORM_MAX_WIDTH)
            speed = self.platform_base_speed * self.np_random.uniform(0.8, 1.2)
            
            plat = {
                "rect": pygame.Rect(x, y, width, self.PLATFORM_HEIGHT),
                "speed": speed, "dir": self.np_random.choice([-1, 1]), "is_goal": False
            }
            self.platforms.append(plat)
            last_y = y
            
        # Goal platform
        goal_plat = self.platforms[-1]
        goal_plat["is_goal"] = True
        goal_plat["rect"].width = self.WIDTH # Make it span the screen
        goal_plat["rect"].x = 0
        goal_plat["speed"] = 0

    def _update_platforms(self):
        for plat in self.platforms:
            if plat["speed"] > 0:
                plat["rect"].y += plat["speed"] * plat["dir"]
                if plat["rect"].top < 0 or plat["rect"].bottom > self.HEIGHT:
                    plat["dir"] *= -1

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.on_ground = False
        reward = 0

        if self.player_vel.y >= 0: # Only collide when moving down
            for plat in self.platforms:
                plat_rect = plat["rect"]
                if player_rect.colliderect(plat_rect):
                    # Check if player's bottom is intersecting the top of the platform
                    if abs(player_rect.bottom - plat_rect.top) < self.MAX_VEL_Y + 1 and player_rect.centerx > plat_rect.left and player_rect.centerx < plat_rect.right:
                        self.player_pos.y = plat_rect.top - self.PLAYER_SIZE
                        self.player_vel.y = 0
                        self.on_ground = True
                        
                        reward += 1 # Base landing reward
                        
                        # Edge landing penalty
                        edge_margin = plat_rect.width * 0.1
                        if player_rect.left < plat_rect.left + edge_margin or player_rect.right > plat_rect.right - edge_margin:
                            reward -= 0.2
                        
                        self._create_landing_particles(pygame.math.Vector2(player_rect.midbottom))

                        if plat["is_goal"]:
                            reward += 100
                            self.score += 100
                            self.game_over = True
                        
                        return reward
        return 0

    def _check_termination(self):
        return (
            self.player_pos.y >= self.HEIGHT or 
            self.time_remaining <= 0 or
            self.game_over
        )

    def _get_observation(self):
        self._draw_background()
        self._render_platforms()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _draw_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = [
                int(self.COLOR_BG_BOTTOM[i] * (1 - ratio) + self.COLOR_BG_TOP[i] * ratio)
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_platforms(self):
        for plat in self.platforms:
            color = self.COLOR_GOAL_PLATFORM if plat["is_goal"] else self.COLOR_PLATFORM
            pygame.draw.rect(self.screen, color, plat["rect"], border_radius=3)
            pygame.draw.rect(self.screen, tuple(int(c*0.8) for c in color), plat["rect"], 1, border_radius=3)

    def _render_player(self):
        player_rect = pygame.Rect(int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_SIZE, self.PLAYER_SIZE)
        # Glow effect
        for i in range(4, 0, -1):
            glow_color = (255, 255, 255, 30 - i * 5)
            s = pygame.Surface((self.PLAYER_SIZE + i*2, self.PLAYER_SIZE + i*2), pygame.SRCALPHA)
            pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=i+2)
            self.screen.blit(s, (player_rect.x - i, player_rect.y - i))

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)

    def _render_ui(self):
        height_score = max(0, int(self.HEIGHT - self.max_height_reached))
        height_text = self.font_large.render(f"Height: {height_score}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (10, 10))

        time_text = self.font_large.render(f"Time: {max(0, int(self.time_remaining))}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)

    def _create_landing_particles(self, pos):
        for _ in range(10):
            vel = pygame.math.Vector2(self.np_random.uniform(-1.5, 1.5), self.np_random.uniform(-1, 0))
            life = self.np_random.integers(15, 30)
            self.particles.append({"pos": pygame.math.Vector2(pos), "vel": vel, "life": life, "max_life": life})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"].y += 0.1 # Particle gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p["life"] / p["max_life"]
            size = int(life_ratio * 4)
            if size > 0:
                color = (
                    self.COLOR_PARTICLE[0],
                    self.COLOR_PARTICLE[1],
                    self.COLOR_PARTICLE[2],
                    int(life_ratio * 255)
                )
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (int(p["pos"].x - size), int(p["pos"].y - size)))

    def _get_info(self):
        height_score = max(0, int(self.HEIGHT - self.max_height_reached))
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "max_height": height_score
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    
    # --- Human Play Setup ---
    # Create a display for human playing
    pygame.display.set_caption("Platform Hopper")
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    total_reward = 0
    
    print("Human Controls:")
    print("  - Left/Right Arrows: Move in air")
    print("  - Space: Large Jump")
    print("  - Shift: Small Jump")
    print("  - Down Arrow: Fall Faster")

    while not done:
        # --- Action Mapping for Human Play ---
        movement = 0 # 0=none
        
        # Get pressed keys
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_LEFT]: movement = 3
        if pressed[pygame.K_RIGHT]: movement = 4
        if pressed[pygame.K_DOWN]: movement = 2
        # UP is not used in this action scheme
        
        space_held = 1 if pressed[pygame.K_SPACE] else 0
        shift_held = 1 if pressed[pygame.K_LSHIFT] or pressed[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Pygame Event Handling for window ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Rendering for Human ---
        # The environment's observation is already a rendered frame
        # We just need to blit it to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()