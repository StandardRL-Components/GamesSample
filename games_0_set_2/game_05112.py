import os
import os
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to set your direction of movement. "
        "Survive as long as you can and reach the green goal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a high-speed square through a procedurally generated neon obstacle field. "
        "Reach the goal before time runs out, but be careful: three collisions and you're out."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (10, 10, 25)
    COLOR_GRID = (30, 30, 50)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200, 100)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_FLASH = (255, 255, 255)
    COLOR_GOAL = (50, 255, 50)
    COLOR_GOAL_PULSE = (150, 255, 150)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_WARN = (255, 50, 50)
    COLOR_TEXT_SPEED = (255, 220, 0)
    
    # Screen and World
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WORLD_WIDTH = 2000
    WORLD_HEIGHT = 2000

    # Game Parameters
    PLAYER_SIZE = 16
    OBSTACLE_SIZE = 24
    GOAL_SIZE = 40
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60 seconds
    MAX_COLLISIONS = 3
    BASE_SPEED = 4.0
    INITIAL_OBSTACLE_COUNT = 40
    INITIAL_GATE_COUNT = 5
    INVINCIBILITY_FRAMES = 15 # 0.5 seconds

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_speed = pygame.font.SysFont("Consolas", 24, bold=True)

        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.obstacles = []
        self.gates = []
        self.goal_rect = pygame.Rect(0,0,0,0)
        
        # self.reset() is called by the agent/wrapper, not needed here.
        # self.validate_implementation() is a debug tool, not for production.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.collision_count = 0
        self.speed_multiplier = 1.0
        self.invincibility_timer = 0
        
        self.player_pos = pygame.Vector2(self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)

        # Procedural Generation
        self._generate_level()
        
        self.last_dist_to_goal = self.player_pos.distance_to(self.goal_rect.center)
        self.passed_gates = set()
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        movement = action[0]
        
        self._update_state(movement)
        self._handle_collisions()
        
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self._check_termination()
        
        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_state(self, movement):
        # Update timers
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

        # Update player velocity based on action
        current_speed = self.BASE_SPEED * self.speed_multiplier
        if movement == 1: # Up
            self.player_vel.y = -current_speed
            self.player_vel.x = 0
        elif movement == 2: # Down
            self.player_vel.y = current_speed
            self.player_vel.x = 0
        elif movement == 3: # Left
            self.player_vel.x = -current_speed
            self.player_vel.y = 0
        elif movement == 4: # Right
            self.player_vel.x = current_speed
            self.player_vel.y = 0
        # movement == 0 (no-op) maintains current velocity

        # Update player position
        self.player_pos += self.player_vel

        # World boundaries
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WORLD_WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.WORLD_HEIGHT)

        # Update difficulty over time
        if self.steps > 0:
            # Increase speed every 10 seconds
            if self.steps % (10 * self.FPS) == 0:
                self.speed_multiplier += 0.1
            # Increase obstacle density every 5 seconds
            if self.steps % (5 * self.FPS) == 0:
                self._add_obstacles(1)

        # Update particles
        self._update_particles()
        if self.player_vel.length() > 0:
            self._create_particle()

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        if self.invincibility_timer > 0:
            return

        for obs in self.obstacles:
            if player_rect.colliderect(obs['rect']):
                # sfx: player_hit
                self.collision_count += 1
                self.speed_multiplier = max(1.0, self.speed_multiplier * 0.8)
                self.invincibility_timer = self.INVINCIBILITY_FRAMES
                obs['flash_timer'] = 5 # Flash for 5 frames
                # Apply a small knockback
                self.player_vel *= -0.5
                break # Only one collision per frame

    def _calculate_reward(self):
        reward = 0.0

        # Distance to goal reward
        current_dist = self.player_pos.distance_to(self.goal_rect.center)
        reward += (self.last_dist_to_goal - current_dist) * 0.01 # Scaled down
        self.last_dist_to_goal = current_dist

        # Collision penalty
        if self.invincibility_timer == self.INVINCIBILITY_FRAMES -1: # Triggered this frame
            reward -= 10.0
        
        # Gate reward
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for i, gate in enumerate(self.gates):
            if i not in self.passed_gates and gate['trigger'].colliderect(player_rect):
                # sfx: gate_passed
                reward += 5.0
                self.passed_gates.add(i)

        # Goal reward
        if player_rect.colliderect(self.goal_rect):
            time_bonus = (self.MAX_STEPS - self.steps) / self.MAX_STEPS * 50
            reward += 100.0 + time_bonus
            # sfx: goal_reached

        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        if player_rect.colliderect(self.goal_rect) or self.collision_count >= self.MAX_COLLISIONS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Camera logic
        camera_offset_x = self.player_pos.x - self.SCREEN_WIDTH / 2
        camera_offset_y = self.player_pos.y - self.SCREEN_HEIGHT / 2

        # Render all game elements
        self._render_background(camera_offset_x, camera_offset_y)
        self._render_particles(camera_offset_x, camera_offset_y)
        self._render_goal(camera_offset_x, camera_offset_y)
        self._render_obstacles(camera_offset_x, camera_offset_y)
        self._render_player(camera_offset_x, camera_offset_y)
        
        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "collisions": self.collision_count,
            "speed_multiplier": self.speed_multiplier,
        }

    # --- Generation and State Helpers ---
    def _generate_level(self):
        self.obstacles = []
        self.gates = []
        
        player_start_area = pygame.Rect(self.WORLD_WIDTH/2 - 100, self.WORLD_HEIGHT/2 - 100, 200, 200)

        # Place goal
        goal_pos = self.player_pos
        while goal_pos.distance_to(self.player_pos) < 800:
            goal_x = self.np_random.integers(self.GOAL_SIZE, self.WORLD_WIDTH - self.GOAL_SIZE)
            goal_y = self.np_random.integers(self.GOAL_SIZE, self.WORLD_HEIGHT - self.GOAL_SIZE)
            goal_pos = pygame.Vector2(goal_x, goal_y)
        self.goal_rect = pygame.Rect(goal_pos.x - self.GOAL_SIZE/2, goal_pos.y - self.GOAL_SIZE/2, self.GOAL_SIZE, self.GOAL_SIZE)

        # Place obstacles
        self._add_obstacles(self.INITIAL_OBSTACLE_COUNT, safe_zone=player_start_area)

        # Place gates
        for _ in range(self.INITIAL_GATE_COUNT):
            self._add_gate(safe_zone=player_start_area)

    def _add_obstacles(self, count, safe_zone=None):
        for _ in range(count):
            while True:
                x = self.np_random.integers(0, self.WORLD_WIDTH - self.OBSTACLE_SIZE)
                y = self.np_random.integers(0, self.WORLD_HEIGHT - self.OBSTACLE_SIZE)
                new_rect = pygame.Rect(x, y, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE)
                
                if (safe_zone and new_rect.colliderect(safe_zone)) or new_rect.colliderect(self.goal_rect):
                    continue
                
                # Check for overlap with existing obstacles
                if not any(new_rect.colliderect(obs['rect']) for obs in self.obstacles):
                    self.obstacles.append({'rect': new_rect, 'flash_timer': 0})
                    break

    def _add_gate(self, safe_zone=None):
        while True:
            gap = self.np_random.uniform(2.5, 4.0) * self.PLAYER_SIZE
            is_horizontal = self.np_random.choice([True, False])
            
            x = self.np_random.integers(0, int(self.WORLD_WIDTH - self.OBSTACLE_SIZE * 2 - gap))
            y = self.np_random.integers(0, int(self.WORLD_HEIGHT - self.OBSTACLE_SIZE * 2 - gap))

            if is_horizontal:
                rect1 = pygame.Rect(x, y, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE)
                rect2 = pygame.Rect(x + self.OBSTACLE_SIZE + gap, y, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE)
                trigger = pygame.Rect(x + self.OBSTACLE_SIZE, y, gap, self.OBSTACLE_SIZE)
            else:
                rect1 = pygame.Rect(x, y, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE)
                rect2 = pygame.Rect(x, y + self.OBSTACLE_SIZE + gap, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE)
                trigger = pygame.Rect(x, y + self.OBSTACLE_SIZE, self.OBSTACLE_SIZE, gap)

            if (safe_zone and (rect1.colliderect(safe_zone) or rect2.colliderect(safe_zone))) or \
               rect1.colliderect(self.goal_rect) or rect2.colliderect(self.goal_rect):
                continue

            if not any(rect1.colliderect(obs['rect']) or rect2.colliderect(obs['rect']) for obs in self.obstacles):
                self.obstacles.append({'rect': rect1, 'flash_timer': 0})
                self.obstacles.append({'rect': rect2, 'flash_timer': 0})
                self.gates.append({'trigger': trigger, 'rects': (rect1, rect2)})
                break

    # --- Particle System ---
    def _create_particle(self):
        self.particles.append({
            'pos': pygame.Vector2(self.player_pos) + pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)),
            'life': 20,
            'size': self.np_random.uniform(3, 7)
        })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['size'] *= 0.95

    # --- Rendering Helpers ---
    def _render_background(self, ox, oy):
        grid_size = 50
        start_x = - (ox % grid_size)
        start_y = - (oy % grid_size)
        for x in range(int(start_x), self.SCREEN_WIDTH, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(int(start_y), self.SCREEN_HEIGHT, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_particles(self, ox, oy):
        for p in self.particles:
            pos = (int(p['pos'].x - ox), int(p['pos'].y - oy))
            size = int(p['size'])
            if size > 0:
                alpha = int(150 * (p['life'] / 20))
                color = (*self.COLOR_PLAYER, alpha)
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.rect(s, color, s.get_rect())
                self.screen.blit(s, (pos[0] - size, pos[1] - size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_goal(self, ox, oy):
        pulse = 1.0 + 0.1 * math.sin(self.steps * 0.1)
        size = int(self.GOAL_SIZE * pulse)
        
        # Draw glow
        glow_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_GOAL, 50), glow_surf.get_rect(), border_radius=int(size*0.3))
        pos_x, pos_y = self.goal_rect.center
        screen_pos = (int(pos_x - ox - size), int(pos_y - oy - size))
        self.screen.blit(glow_surf, screen_pos, special_flags=pygame.BLEND_RGBA_ADD)

        # Draw main rect
        rect = pygame.Rect(0,0,size,size)
        rect.center = (pos_x - ox, pos_y - oy)
        pygame.draw.rect(self.screen, self.COLOR_GOAL_PULSE, rect, border_radius=int(size*0.2))

    def _render_obstacles(self, ox, oy):
        for obs in self.obstacles:
            rect = obs['rect']
            screen_rect = rect.move(-ox, -oy)
            
            if self.screen.get_rect().colliderect(screen_rect):
                color = self.COLOR_OBSTACLE_FLASH if obs['flash_timer'] > 0 else self.COLOR_OBSTACLE
                if obs['flash_timer'] > 0:
                    obs['flash_timer'] -= 1
                pygame.draw.rect(self.screen, color, screen_rect, border_radius=4)

    def _render_player(self, ox, oy):
        player_screen_pos = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        # Invincibility flash
        if self.invincibility_timer > 0 and self.steps % 4 < 2:
            return

        # Glow
        glow_radius = int(self.PLAYER_SIZE * 1.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (player_screen_pos[0] - glow_radius, player_screen_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Main body
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = player_screen_pos
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
    def _render_ui(self):
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = f"TIME: {max(0, time_left):.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (10, 10))

        # Collisions
        collision_text = f"HITS: {self.collision_count}/{self.MAX_COLLISIONS}"
        collision_surf = self.font_ui.render(collision_text, True, self.COLOR_TEXT_WARN)
        self.screen.blit(collision_surf, (self.SCREEN_WIDTH - collision_surf.get_width() - 10, 10))

        # Speed Multiplier
        speed_text = f"{self.speed_multiplier:.1f}x"
        speed_surf = self.font_speed.render(speed_text, True, self.COLOR_TEXT_SPEED)
        pos = (self.SCREEN_WIDTH / 2 - speed_surf.get_width() / 2, self.SCREEN_HEIGHT - speed_surf.get_height() - 10)
        self.screen.blit(speed_surf, pos)

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # This environment is configured for headless execution by default.
    # To play manually, you would need to comment out the os.environ line at the top of the file
    # and ensure you have a display and pygame installed.
    
    env = GameEnv()
    
    try:
        obs, info = env.reset()
        running = True
        is_human_play = "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] != "dummy"
        
        if is_human_play:
            render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
            pygame.display.set_caption(env.game_description)
        else:
            print("Running in headless mode.")

        action = env.action_space.sample()
        action[0] = 0 # Start with no-op for movement

        while running:
            movement_action = 0 # No-op by default
            
            if is_human_play:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]: movement_action = 1
                elif keys[pygame.K_DOWN]: movement_action = 2
                elif keys[pygame.K_LEFT]: movement_action = 3
                elif keys[pygame.K_RIGHT]: movement_action = 4
                
                action[0] = movement_action
            else: # Simple agent for headless mode
                if env.steps % 60 == 0: # Change direction every 2 seconds
                    action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"Episode finished in {info['steps']} steps. Final Score: {info['score']:.2f}")
                obs, info = env.reset()

            if is_human_play:
                # Transpose back for pygame display
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                render_screen.blit(surf, (0, 0))
                pygame.display.flip()

    finally:
        env.close()