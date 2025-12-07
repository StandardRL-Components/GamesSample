import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Butterfly Forest Racer

    A Gymnasium environment where an agent controls a genetically modified butterfly
    racing through a procedurally generated forest. The core mechanics involve
    flipping gravity to navigate obstacles and teleporting between glowing flowers
    for speed boosts.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up-gravity, 2=down-gravity, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held for teleport)
    - actions[2]: Shift button (0=released, 1=held, unused)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +0.01 per step for forward progress.
    - +5.0 for teleporting to a flower.
    - -0.1 for being close to an obstacle.
    - +100 for finishing the level, scaled by remaining time.
    - -100 for colliding with an obstacle.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Race a butterfly through a forest, flipping gravity to avoid branches "
        "and teleporting between flowers for a speed boost."
    )
    user_guide = (
        "Controls: ↑ to flip gravity up, ↓ to flip gravity down. "
        "Press space to teleport to the nearest flower."
    )
    auto_advance = True


    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    LEVEL_LENGTH = 6400  # 10 screens long
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (12, 20, 39)
    COLOR_BUTTERFLY = (255, 200, 50)
    COLOR_BUTTERFLY_BOOST = (255, 255, 100)
    COLOR_OBSTACLE = (40, 80, 60)
    COLOR_OBSTACLE_TRUNK = (30, 60, 50)
    COLOR_FLOWER = (50, 150, 255)
    COLOR_FLOWER_GLOW = (100, 200, 255)
    COLOR_PARTICLE = (200, 220, 255)
    COLOR_UI_TEXT = (230, 230, 240)
    COLOR_UI_BG = (20, 40, 70, 180)
    COLOR_PROGRESS_BAR = (100, 200, 255)
    COLOR_PROGRESS_BAR_BG = (40, 80, 120)

    # Butterfly Physics
    BUTTERFLY_BASE_SPEED = 3.0
    BUTTERFLY_BOOST_SPEED = 6.0
    BUTTERFLY_GRAVITY = 0.25
    BUTTERFLY_SIZE = 12
    BOOST_DURATION = 90  # frames
    TELEPORT_COOLDOWN_TIME = 30 # frames

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.clock = pygame.time.Clock()

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.butterfly_pos = pygame.Vector2(0, 0)
        self.butterfly_vel = pygame.Vector2(0, 0)
        self.gravity_direction = 1
        self.boost_timer = 0
        self.teleport_cooldown = 0
        
        self.camera_x = 0
        self.branches = []
        self.flowers = []
        self.particles = []
        self.parallax_bg = []
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.butterfly_pos = pygame.Vector2(100, self.SCREEN_HEIGHT / 2)
        self.butterfly_vel = pygame.Vector2(self.BUTTERFLY_BASE_SPEED, 0)
        self.gravity_direction = 1
        self.boost_timer = 0
        self.teleport_cooldown = 0
        
        self.camera_x = 0
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # On subsequent steps after termination, return the final state
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Handle Actions ---
        movement_action = action[0]
        teleport_action = action[1] == 1

        if movement_action == 1:  # Set gravity up
            self.gravity_direction = -1
        elif movement_action == 2:  # Set gravity down
            self.gravity_direction = 1
        
        if self.teleport_cooldown > 0:
            self.teleport_cooldown -= 1

        if teleport_action and self.teleport_cooldown == 0:
            reward += self._handle_teleport()

        # --- Update Game State ---
        self._update_butterfly()
        self._update_particles()
        
        self.camera_x = self.butterfly_pos.x - 100

        # --- Collision and Termination ---
        collision_detected, near_miss = self._check_collisions()
        
        terminated = False
        truncated = False
        if collision_detected:
            reward = -100.0
            self._create_particles(self.butterfly_pos, 50, (255, 50, 50), 4) # Sound: explosion
            terminated = True
            self.game_over = True
        elif self.butterfly_pos.x >= self.LEVEL_LENGTH:
            time_bonus = (self.MAX_STEPS - self.steps) / self.MAX_STEPS
            reward = 100.0 + 100.0 * max(0, time_bonus)
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward = -10.0 # Penalty for timeout
            terminated = True # Using terminated for timeout as it's a losing condition
            self.game_over = True
        
        # --- Calculate Reward ---
        if not terminated:
            # Base reward for forward progress
            reward += 0.01 * (self.butterfly_vel.x / self.BUTTERFLY_BASE_SPEED)
            if near_miss:
                reward -= 0.1

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_level(self):
        self.branches = []
        self.flowers = []
        self.particles = []
        
        # Generate parallax background elements
        self.parallax_bg = []
        for _ in range(100):
            # Layer 1 (slow)
            self.parallax_bg.append({
                "pos": pygame.Vector2(random.uniform(0, self.LEVEL_LENGTH), random.uniform(0, self.SCREEN_HEIGHT)),
                "depth": 0.2,
                "radius": random.randint(1, 2)
            })
            # Layer 2 (fast)
            self.parallax_bg.append({
                "pos": pygame.Vector2(random.uniform(0, self.LEVEL_LENGTH), random.uniform(0, self.SCREEN_HEIGHT)),
                "depth": 0.5,
                "radius": random.randint(1, 3)
            })

        num_segments = int(self.LEVEL_LENGTH / 400)
        min_gap = 180 # Ensure path is navigable

        for i in range(1, num_segments):
            segment_x = i * 400
            
            # Place obstacles (branches)
            obstacle_y = random.uniform(0, self.SCREEN_HEIGHT - min_gap)
            
            branch_width = random.randint(20, 50)

            if random.random() < 0.7: # Branch from top and bottom
                self.branches.append(pygame.Rect(segment_x + random.uniform(-50, 50), 0, branch_width, obstacle_y))
                self.branches.append(pygame.Rect(segment_x + random.uniform(-50, 50), obstacle_y + min_gap, branch_width, self.SCREEN_HEIGHT - (obstacle_y + min_gap)))
            else: # Branch from sides (less common)
                 self.branches.append(pygame.Rect(segment_x + random.uniform(-50, 50), 0, branch_width, self.SCREEN_HEIGHT - (obstacle_y + min_gap)))
                 self.branches.append(pygame.Rect(segment_x + random.uniform(-50, 50), self.SCREEN_HEIGHT-obstacle_y, branch_width, obstacle_y))


            # Place a flower in the gap
            if random.random() < 0.6:
                flower_x = segment_x + random.uniform(100, 300)
                flower_y = obstacle_y + min_gap / 2 + random.uniform(-20, 20)
                self.flowers.append({"pos": pygame.Vector2(flower_x, flower_y), "active": True, "radius": 10})
    
    def _handle_teleport(self):
        # Find nearest flower in front of the butterfly
        best_flower = None
        min_dist_sq = float('inf')
        
        for flower in self.flowers:
            if flower["active"] and flower["pos"].x > self.butterfly_pos.x:
                dist_sq = self.butterfly_pos.distance_squared_to(flower["pos"])
                if dist_sq < min_dist_sq and dist_sq < (300**2): # Max teleport range
                    min_dist_sq = dist_sq
                    best_flower = flower
        
        if best_flower:
            # Sound: teleport_whoosh
            # Create particles at old and new locations
            self._create_particles(self.butterfly_pos, 30, self.COLOR_FLOWER, 2)
            self.butterfly_pos.x, self.butterfly_pos.y = best_flower["pos"].x, best_flower["pos"].y
            self._create_particles(self.butterfly_pos, 50, self.COLOR_FLOWER, 3)
            
            best_flower["active"] = False
            self.boost_timer = self.BOOST_DURATION
            self.teleport_cooldown = self.TELEPORT_COOLDOWN_TIME
            return 5.0 # Reward for teleporting
        
        return 0.0

    def _update_butterfly(self):
        # Update speed based on boost
        if self.boost_timer > 0:
            self.boost_timer -= 1
            current_speed = self.BUTTERFLY_BOOST_SPEED
        else:
            current_speed = self.BUTTERFLY_BASE_SPEED
        
        self.butterfly_vel.x = current_speed
        
        # Apply gravity
        self.butterfly_vel.y += self.gravity_direction * self.BUTTERFLY_GRAVITY
        
        # Update position
        self.butterfly_pos += self.butterfly_vel

        # Clamp position to screen bounds (vertically)
        if self.butterfly_pos.y < self.BUTTERFLY_SIZE:
            self.butterfly_pos.y = self.BUTTERFLY_SIZE
            self.butterfly_vel.y = 0
        if self.butterfly_pos.y > self.SCREEN_HEIGHT - self.BUTTERFLY_SIZE:
            self.butterfly_pos.y = self.SCREEN_HEIGHT - self.BUTTERFLY_SIZE
            self.butterfly_vel.y = 0
            
        # Add trail particles
        if self.steps % 3 == 0:
            p_color = self.COLOR_BUTTERFLY_BOOST if self.boost_timer > 0 else self.COLOR_BUTTERFLY
            self.particles.append({
                "pos": pygame.Vector2(self.butterfly_pos),
                "vel": pygame.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)),
                "lifetime": 20,
                "color": p_color,
                "radius": random.randint(1, 3)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifetime"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifetime"] -= 1

    def _check_collisions(self):
        butterfly_rect = pygame.Rect(
            self.butterfly_pos.x - self.BUTTERFLY_SIZE / 2,
            self.butterfly_pos.y - self.BUTTERFLY_SIZE / 2,
            self.BUTTERFLY_SIZE, self.BUTTERFLY_SIZE
        )
        near_miss = False
        near_miss_dist = self.BUTTERFLY_SIZE * 4

        for branch in self.branches:
            if butterfly_rect.colliderect(branch):
                return True, False
            
            # Check for near miss
            clamped_x = max(branch.left, min(self.butterfly_pos.x, branch.right))
            clamped_y = max(branch.top, min(self.butterfly_pos.y, branch.bottom))
            dist = self.butterfly_pos.distance_to(pygame.Vector2(clamped_x, clamped_y))
            if dist < near_miss_dist:
                near_miss = True

        return False, near_miss

    def _create_particles(self, pos, count, color, speed_mult):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifetime": random.randint(20, 40),
                "color": color,
                "radius": random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render parallax background
        for bg_item in self.parallax_bg:
            screen_pos_x = (bg_item["pos"].x - self.camera_x * bg_item["depth"]) % self.SCREEN_WIDTH
            pygame.gfxdraw.filled_circle(
                self.screen, 
                int(screen_pos_x), 
                int(bg_item["pos"].y), 
                bg_item["radius"], 
                (30, 45, 70)
            )

        # Render branches
        for branch in self.branches:
            screen_rect = branch.move(-self.camera_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)

        # Render flowers
        for flower in self.flowers:
            if flower["active"]:
                screen_pos = flower["pos"] - pygame.Vector2(self.camera_x, 0)
                if -20 < screen_pos.x < self.SCREEN_WIDTH + 20:
                    # Glow effect
                    glow_radius = int(flower["radius"] * (1.5 + 0.5 * math.sin(self.steps * 0.1)))
                    self._draw_glow(screen_pos, self.COLOR_FLOWER_GLOW, glow_radius)
                    # Core flower
                    pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), flower["radius"], self.COLOR_FLOWER)
                    pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), flower["radius"], self.COLOR_FLOWER_GLOW)
    
        # Render particles
        for p in self.particles:
            screen_pos = p["pos"] - pygame.Vector2(self.camera_x, 0)
            alpha = int(255 * (p["lifetime"] / 40.0))
            color = (*p["color"], alpha)
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), int(p["radius"]), color)
            except TypeError: # Handle potential color with alpha issues
                pygame.draw.circle(self.screen, p["color"], (int(screen_pos.x), int(screen_pos.y)), int(p["radius"]))
            
        # Render butterfly
        self._render_butterfly()
        
        # Render finish line
        finish_x = self.LEVEL_LENGTH - self.camera_x
        if finish_x < self.SCREEN_WIDTH:
            pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (finish_x, 0), (finish_x, self.SCREEN_HEIGHT), 3)
            finish_text = self.font_large.render("FINISH", True, self.COLOR_UI_TEXT)
            self.screen.blit(finish_text, (finish_x + 10, self.SCREEN_HEIGHT / 2 - 20))


    def _render_butterfly(self):
        screen_pos = self.butterfly_pos - pygame.Vector2(self.camera_x, 0)
        
        is_boosting = self.boost_timer > 0
        b_color = self.COLOR_BUTTERFLY_BOOST if is_boosting else self.COLOR_BUTTERFLY
        
        # Glow
        self._draw_glow(screen_pos, b_color, int(self.BUTTERFLY_SIZE * (1.8 if is_boosting else 1.5)))

        # Body
        pygame.draw.circle(self.screen, b_color, (int(screen_pos.x), int(screen_pos.y)), int(self.BUTTERFLY_SIZE / 2))

        # Wings
        wing_angle = math.sin(self.steps * 0.4) * 0.6  # Radians
        wing_length = self.BUTTERFLY_SIZE * 1.5
        
        p1 = screen_pos + pygame.Vector2(wing_length, 0).rotate_rad(wing_angle)
        p2 = screen_pos + pygame.Vector2(wing_length, 0).rotate_rad(-wing_angle)
        p3 = screen_pos + pygame.Vector2(wing_length/2, 0).rotate_rad(wing_angle + math.pi/2)
        p4 = screen_pos + pygame.Vector2(wing_length/2, 0).rotate_rad(-wing_angle - math.pi/2)
        
        pygame.gfxdraw.aapolygon(self.screen, [(screen_pos.x, screen_pos.y), (p1.x, p1.y), (p3.x, p3.y)], b_color)
        pygame.gfxdraw.filled_polygon(self.screen, [(screen_pos.x, screen_pos.y), (p1.x, p1.y), (p3.x, p3.y)], b_color)
        pygame.gfxdraw.aapolygon(self.screen, [(screen_pos.x, screen_pos.y), (p2.x, p2.y), (p4.x, p4.y)], b_color)
        pygame.gfxdraw.filled_polygon(self.screen, [(screen_pos.x, screen_pos.y), (p2.x, p2.y), (p4.x, p4.y)], b_color)

        # Gravity indicator
        arrow_y_offset = (self.BUTTERFLY_SIZE + 5) * self.gravity_direction
        arrow_start = (int(screen_pos.x), int(screen_pos.y + arrow_y_offset))
        arrow_end = (int(screen_pos.x), int(screen_pos.y + arrow_y_offset + 5 * self.gravity_direction))
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, arrow_start, arrow_end, 2)

    def _draw_glow(self, pos, color, radius):
        for i in range(radius, 0, -2):
            alpha = 40 * (1 - i / radius)
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), i, (*color, int(alpha)))
            except (TypeError, ValueError): # Fallback if color has alpha already or other issue
                pass


    def _render_ui(self):
        # Semi-transparent background for UI
        ui_surface = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        ui_surface.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surface, (0, 0))
        
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_text = self.font_small.render(f"TIME: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 15))
        
        # Boost bar
        if self.boost_timer > 0:
            boost_ratio = self.boost_timer / self.BOOST_DURATION
            bar_width = 100
            pygame.draw.rect(self.screen, self.COLOR_BUTTERFLY_BOOST, (10, 35, bar_width * boost_ratio, 8))
        
        # Progress bar
        progress = self.butterfly_pos.x / self.LEVEL_LENGTH
        progress = max(0, min(1, progress))
        bar_width = self.SCREEN_WIDTH - 20
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_BG, (10, self.SCREEN_HEIGHT - 15, bar_width, 10))
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR, (10, self.SCREEN_HEIGHT - 15, bar_width * progress, 10))
        
        if self.game_over:
            if self.butterfly_pos.x >= self.LEVEL_LENGTH:
                end_text = "LEVEL COMPLETE"
            else:
                end_text = "GAME OVER"
            end_surf = self.font_large.render(end_text, True, self.COLOR_UI_TEXT)
            self.screen.blit(end_surf, (self.SCREEN_WIDTH/2 - end_surf.get_width()/2, self.SCREEN_HEIGHT/2 - end_surf.get_height()/2))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "butterfly_pos": (self.butterfly_pos.x, self.butterfly_pos.y),
            "progress_pct": self.butterfly_pos.x / self.LEVEL_LENGTH
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Use a display for manual play
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Butterfly Forest Racer")
    clock = pygame.time.Clock()

    while running:
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        # Construct the action
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000) # Pause before restarting

        clock.tick(30) # Run at 30 FPS

    env.close()