import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: On a platform, use ↑,↓,←,→ + SPACE to jump. In the air, use ←,→ + SHIFT for a thruster boost."
    )

    game_description = (
        "A side-view arcade game where you control a hopping spaceship. Reach the top platform while managing fuel. Land on platforms to refuel."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (10, 10, 42)
    COLOR_PLAYER = (64, 224, 208)
    COLOR_PLAYER_GLOW = (64, 224, 208, 50)
    COLOR_PLATFORM = (255, 255, 255)
    COLOR_GOAL_PLATFORM = (255, 215, 0)
    COLOR_PARTICLE = (255, 165, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_FUEL_HIGH = (0, 255, 0)
    COLOR_FUEL_LOW = (255, 0, 0)

    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Physics
    GRAVITY = 0.4
    AIR_DRAG = 0.99
    MAX_FUEL = 100.0
    FUEL_REPLENISH_ON_LAND = 25.0
    JUMP_FUEL_COST = {0: 8, 1: 12, 2: 5, 3: 10, 4: 10}
    THRUST_FUEL_COST = 1.0

    # Game
    MAX_STEPS = 2000
    NUM_STARS = 150
    WORLD_WIDTH_FACTOR = 2
    WORLD_HEIGHT_SCALAR = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_on_platform = True
        self.player_size = pygame.Vector2(20, 25)
        
        self.camera_pos = pygame.Vector2(0, 0)
        self.platforms = []
        self.stars = []
        self.particles = deque()

        self.fuel = 0.0
        self.score = 0
        self.steps = 0
        self.stage = 1
        self.game_over = False
        
        # self.reset() is called by the wrapper, no need to call it here.
        # However, to allow instantiation without a wrapper, we may need to initialize state.
        # The traceback shows it's called, so we'll keep it, but ensure state is ready.
        self._initialize_state()
        
    def _initialize_state(self):
        """Initializes all state variables to default values before the first reset."""
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_on_platform = True
        
        self.fuel = self.MAX_FUEL
        self.score = 0
        self.steps = 0
        self.stage = 1
        self.game_over = False
        
        self.particles.clear()
        self._generate_stars()
        self._generate_platforms()

        # Center camera on player start
        self.camera_pos = pygame.Vector2(
            self.player_pos.x - self.SCREEN_WIDTH / 2,
            self.player_pos.y - self.SCREEN_HEIGHT * 0.8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.01  # Survival reward

        # --- Handle Action ---
        action_reward = self._handle_action(action)
        reward += action_reward

        # --- Update Physics ---
        self._update_physics()

        # --- Check Collisions ---
        collision_reward = self._check_collisions()
        reward += collision_reward

        # --- Update Game State ---
        self._update_particles()
        self._update_camera()

        # --- Check Termination ---
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        if terminated:
            self.game_over = True

        self.score += reward
        self.steps += 1
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Gymnasium standard is to set terminated=True on truncation

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Jump action (only on platform)
        if self.player_on_platform and space_held and self.fuel > 0:
            fuel_cost = self.JUMP_FUEL_COST.get(movement, 8)
            if self.fuel >= fuel_cost:
                self.player_on_platform = False
                # Sound: jump.wav
                if movement == 1:  # Up
                    self.player_vel = pygame.Vector2(0, -11)
                elif movement == 2:  # Down (hop)
                    self.player_vel = pygame.Vector2(0, -6)
                elif movement == 3:  # Left
                    self.player_vel = pygame.Vector2(-6, -9)
                elif movement == 4:  # Right
                    self.player_vel = pygame.Vector2(6, -9)
                else:  # None
                    self.player_vel = pygame.Vector2(0, -9)
                
                self.fuel -= fuel_cost
                reward -= 0.2 * fuel_cost # Brief: -0.2 per fuel unit

        # Thruster action (only in air)
        elif not self.player_on_platform and shift_held and self.fuel > 0:
            thrust_dir = 0
            if movement == 3: # Left
                thrust_dir = -1
            elif movement == 4: # Right
                thrust_dir = 1
            
            if thrust_dir != 0:
                self.player_vel.x += thrust_dir * 0.5
                self.fuel -= self.THRUST_FUEL_COST
                reward -= 0.2 * self.THRUST_FUEL_COST
                # Sound: thrust.wav
                # Add particles
                for _ in range(2):
                    p_vel = pygame.Vector2(-thrust_dir * self.np_random.uniform(2, 4), self.np_random.uniform(-1, 1))
                    self.particles.append({
                        'pos': self.player_pos.copy() + pygame.Vector2(self.player_size.x/2, self.player_size.y/2),
                        'vel': p_vel,
                        'life': self.np_random.integers(15, 25)
                    })

        return reward

    def _update_physics(self):
        if not self.player_on_platform:
            self.player_vel.y += self.GRAVITY
        
        self.player_vel.x *= self.AIR_DRAG
        self.player_pos += self.player_vel

    def _check_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos, self.player_size)

        # Falling down and player feet are at the collision point
        if self.player_vel.y > 0:
            for i, plat in enumerate(self.platforms):
                if player_rect.colliderect(plat) and player_rect.bottom < plat.bottom:
                    # Check if the player's previous bottom was above the platform's top
                    if (self.player_pos.y - self.player_vel.y + self.player_size.y) <= plat.top:
                        self.player_pos.y = plat.top - self.player_size.y
                        self.player_vel.y = 0
                        self.player_vel.x = 0 # Stop horizontal movement on land
                        self.player_on_platform = True
                        
                        # Sound: land.wav
                        self.fuel = min(self.MAX_FUEL, self.fuel + self.FUEL_REPLENISH_ON_LAND)
                        
                        reward += 1 # Brief: +1 for landing
                        
                        # Risky jump reward
                        player_center_x = player_rect.centerx
                        plat_edge_dist = min(player_center_x - plat.left, plat.right - player_center_x)
                        if plat_edge_dist < plat.width * 0.15: # Landed on outer 15%
                            reward += 5 # Brief: +5 for risky jump

                        # Check for stage progression
                        new_stage = self._get_stage_from_y(self.player_pos.y)
                        if new_stage > self.stage:
                            self.stage = new_stage
                        
                        # Check for win condition
                        if i == len(self.platforms) - 1:
                            self.game_over = True # Handled in termination check
                        
                        break
        return reward
    
    def _check_termination(self):
        terminated = False
        reward = 0

        # Win condition: landed on the last platform
        player_rect = pygame.Rect(self.player_pos, self.player_size)
        goal_platform = self.platforms[-1]
        if self.player_on_platform and player_rect.colliderect(goal_platform):
            terminated = True
            reward = 100 # Brief: +100 for win
            # Sound: win.wav
        
        # Lose condition: fall off screen
        if self.player_pos.y > self.camera_pos.y + self.SCREEN_HEIGHT + 50:
            terminated = True
            reward = -100 # Brief: -100 for fall
            # Sound: lose.wav

        # Lose condition: run out of fuel in mid-air
        if self.fuel <= 0 and not self.player_on_platform:
            terminated = True
            reward = -100 # Brief: -100 for no fuel
            # Sound: lose.wav

        # Max steps is handled by `truncated` in the step function
        
        return terminated, reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_stars()
        self._draw_platforms()
        self._draw_particles()
        self._draw_player()

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Fuel Gauge
        fuel_percent = max(0, self.fuel / self.MAX_FUEL)
        fuel_bar_width = 150
        fuel_bar_height = 15
        fuel_bar_x = self.SCREEN_WIDTH - fuel_bar_width - 10
        fuel_bar_y = 10
        
        # Interpolate color from green to red
        fuel_color = (
            self.COLOR_FUEL_LOW[0] * (1 - fuel_percent) + self.COLOR_FUEL_HIGH[0] * fuel_percent,
            self.COLOR_FUEL_LOW[1] * (1 - fuel_percent) + self.COLOR_FUEL_HIGH[1] * fuel_percent,
            self.COLOR_FUEL_LOW[2] * (1 - fuel_percent) + self.COLOR_FUEL_HIGH[2] * fuel_percent,
        )

        pygame.draw.rect(self.screen, (50, 50, 50), (fuel_bar_x, fuel_bar_y, fuel_bar_width, fuel_bar_height))
        pygame.draw.rect(self.screen, fuel_color, (fuel_bar_x, fuel_bar_y, int(fuel_bar_width * fuel_percent), fuel_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (fuel_bar_x, fuel_bar_y, fuel_bar_width, fuel_bar_height), 1)

        # Stage
        stage_text = self.font_large.render(f"STAGE {self.stage}", True, self.COLOR_TEXT)
        text_rect = stage_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20))
        self.screen.blit(stage_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.fuel,
            "stage": self.stage,
        }

    def _generate_platforms(self):
        self.platforms.clear()
        
        # Start platform
        start_plat_y = self.player_pos.y + self.player_size.y + 10
        start_plat = pygame.Rect(self.player_pos.x - 50, start_plat_y, 150, 20)
        self.platforms.append(start_plat)
        
        current_y = start_plat.y
        world_width = self.SCREEN_WIDTH * self.WORLD_WIDTH_FACTOR
        world_height = self.SCREEN_HEIGHT * self.WORLD_HEIGHT_SCALAR
        
        num_platforms = 0
        # Generate platforms until we reach the top of the world
        while current_y > -world_height and num_platforms < 200: # Safety break
            stage = self._get_stage_from_y(current_y)
            
            # Difficulty scaling
            plat_width = max(60, 120 - (stage - 1) * 20) # 120 -> 100 -> 80
            gap_y_min = 120 + (stage - 1) * 20
            gap_y_max = 180 + (stage - 1) * 30
            gap_x_max = 100 + (stage - 1) * 20
            
            last_plat = self.platforms[-1]
            
            dy = self.np_random.uniform(gap_y_min, gap_y_max)
            dx = self.np_random.uniform(-gap_x_max, gap_x_max)
            
            new_x = last_plat.centerx + dx
            # Clamp to world bounds
            new_x = max(plat_width / 2, min(world_width - plat_width / 2, new_x))
            
            new_y = current_y - dy
            
            plat = pygame.Rect(new_x - plat_width / 2, new_y, plat_width, 20)
            self.platforms.append(plat)
            current_y = new_y
            num_platforms += 1
        
        # Goal platform
        goal_plat = self.platforms.pop() # Remove last generated one
        goal_y = goal_plat.y
        goal_x = self.np_random.uniform(world_width * 0.25, world_width * 0.75)
        self.platforms.append(pygame.Rect(goal_x - 100, goal_y, 200, 30))

    def _get_stage_from_y(self, y_pos):
        if y_pos > -self.SCREEN_HEIGHT * 3:
            return 1
        elif y_pos > -self.SCREEN_HEIGHT * 6:
            return 2
        else:
            return 3

    def _generate_stars(self):
        self.stars.clear()
        world_width = self.SCREEN_WIDTH * self.WORLD_WIDTH_FACTOR
        world_height = self.SCREEN_HEIGHT * self.WORLD_HEIGHT_SCALAR
        for _ in range(self.NUM_STARS):
            self.stars.append({
                'pos': pygame.Vector2(
                    self.np_random.uniform(0, world_width),
                    self.np_random.uniform(-world_height, self.SCREEN_HEIGHT)
                ),
                'size': self.np_random.uniform(0.5, 2),
                'layer': self.np_random.choice([0.2, 0.5, 0.8]) # Parallax layers
            })

    def _update_camera(self):
        target_cam_x = self.player_pos.x - self.SCREEN_WIDTH / 2
        target_cam_y = self.player_pos.y - self.SCREEN_HEIGHT * 0.6
        
        # Smooth camera follow
        self.camera_pos.x = self.camera_pos.x * 0.9 + target_cam_x * 0.1
        self.camera_pos.y = self.camera_pos.y * 0.9 + target_cam_y * 0.1

    def _draw_player(self):
        # Screen position
        screen_pos = self.player_pos - self.camera_pos
        
        # Glow effect
        glow_radius = int(self.player_size.x * 1.2)
        pygame.gfxdraw.filled_circle(
            self.screen, int(screen_pos.x + self.player_size.x / 2), int(screen_pos.y + self.player_size.y / 2),
            glow_radius, self.COLOR_PLAYER_GLOW
        )
        
        # Main body (triangle)
        p1 = (screen_pos.x + self.player_size.x / 2, screen_pos.y)
        p2 = (screen_pos.x, screen_pos.y + self.player_size.y)
        p3 = (screen_pos.x + self.player_size.x, screen_pos.y + self.player_size.y)
        
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3])
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

    def _draw_platforms(self):
        for i, plat in enumerate(self.platforms):
            screen_rect = plat.move(-self.camera_pos.x, -self.camera_pos.y)
            if screen_rect.colliderect(self.screen.get_rect()):
                color = self.COLOR_GOAL_PLATFORM if i == len(self.platforms) - 1 else self.COLOR_PLATFORM
                pygame.draw.rect(self.screen, color, screen_rect, border_radius=3)

    def _draw_stars(self):
        for star in self.stars:
            screen_pos_x = (star['pos'].x - self.camera_pos.x * star['layer'])
            screen_pos_y = (star['pos'].y - self.camera_pos.y * star['layer'])

            # Wrap stars horizontally to create an infinite feel
            world_render_width = self.SCREEN_WIDTH / star['layer']
            screen_pos_x = screen_pos_x % world_render_width

            if 0 <= screen_pos_x < self.SCREEN_WIDTH and 0 <= screen_pos_y < self.SCREEN_HEIGHT:
                size = int(star['size'])
                color_val = int(100 + 100 * star['layer'])
                color = (color_val, color_val, color_val + 50)
                if size < 2:
                    self.screen.set_at((int(screen_pos_x), int(screen_pos_y)), color)
                else:
                    pygame.draw.circle(self.screen, color, (int(screen_pos_x), int(screen_pos_y)), size)

    def _update_particles(self):
        for _ in range(len(self.particles)):
            p = self.particles.popleft()
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] > 0:
                self.particles.append(p)
    
    def _draw_particles(self):
        for p in self.particles:
            screen_pos = p['pos'] - self.camera_pos
            alpha = max(0, 255 * (p['life'] / 25))
            color = (*self.COLOR_PARTICLE, alpha)
            
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(screen_pos.x - 2), int(screen_pos.y - 2)))
            
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a headless environment
    # To see the game, run this file locally with a display
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Lunar Hopper")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)

    while running:
        movement = 0 # None
        space = 0 # Released
        shift = 0 # Released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting

        clock.tick(30) # Match the intended FPS
        
    env.close()