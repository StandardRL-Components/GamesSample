import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:01:38.004802
# Source Brief: brief_00225.md
# Brief Index: 225
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Guide your flock of birds through a sky full of obstacles. Collect pellets to increase your speed and survive as long as possible."
    user_guide = "Controls: Use the arrow keys (↑↓←→) to guide the lead bird. The rest of the flock will follow."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30  # Logic and rendering FPS

    # Colors
    COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
    COLOR_BG_BOTTOM = (70, 130, 180)  # Steel Blue
    COLOR_BIRD = (255, 255, 255)
    COLOR_BIRD_GLOW = (255, 255, 255, 50)
    COLOR_OBSTACLE = (220, 20, 60)  # Crimson
    COLOR_OBSTACLE_GLOW = (220, 20, 60, 50)
    COLOR_PELLET = (255, 255, 0)  # Yellow
    COLOR_PELLET_GLOW = (255, 255, 0, 70)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0, 100)
    
    # Game Parameters
    INITIAL_BIRDS = 5
    MAX_LEVEL = 8
    LEVEL_DURATION_SECONDS = 60
    MAX_EPISODE_STEPS = 5000

    # Bird Physics
    BIRD_ACCELERATION = 1.5
    BIRD_FRICTION = 0.92
    BIRD_MAX_SPEED = 8.0
    BIRD_FLAP_SPEED = 0.5
    BIRD_FLAP_AMP = 3
    BIRD_SIZE = 10
    FLOCK_FORMATION_DISTANCE = 30
    FLOCK_COHESION_STRENGTH = 0.05

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 16)
        
        # State variables are initialized in reset()
        self.birds = []
        self.obstacles = []
        self.pellets = []
        self.particles = []
        
        # Initialize state
        # self.reset() is called by the wrapper/runner

    def _init_level_parameters(self):
        """Sets game parameters based on the current level."""
        level_factor = self.level - 1
        self.obstacle_speed = 3.0 * (1.03 ** level_factor)
        self.obstacle_spawn_interval = max(10, 45 - level_factor * 3) # More frequent spawns
        self.pellet_spawn_interval = int(20 * (1.01 ** level_factor))
        
        self.obstacle_spawn_timer = 0
        self.pellet_spawn_timer = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.level_timer = self.LEVEL_DURATION_SECONDS * self.FPS
        self.total_pellets_collected = 0
        self.flock_speed_multiplier = 1.0

        self._init_level_parameters()

        # Create flock
        self.birds = []
        center_x, center_y = self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 3
        for i in range(self.INITIAL_BIRDS):
            # Arrange in a V-formation initially
            offset_x = (i % 2 * 2 - 1) * (math.ceil(i/2) * self.FLOCK_FORMATION_DISTANCE) if i > 0 else 0
            offset_y = math.ceil(i/2) * self.FLOCK_FORMATION_DISTANCE if i > 0 else 0
            self.birds.append({
                "pos": pygame.Vector2(center_x + offset_x, center_y + offset_y),
                "vel": pygame.Vector2(0, 0),
                "flap_offset": self.np_random.random() * math.pi * 2
            })
        
        self.obstacles = []
        self.pellets = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.1  # Survival reward per frame

        # --- Update game logic ---
        self._handle_input(action)
        self._update_flock()
        self._update_obstacles()
        self._update_pellets_and_particles()
        
        reward += self._handle_collisions()
        self._spawn_entities()

        # --- Update timers and progression ---
        self.steps += 1
        self.level_timer -= 1

        if self.level_timer <= 0:
            if self.level < self.MAX_LEVEL:
                self.level += 1
                self.level_timer = self.LEVEL_DURATION_SECONDS * self.FPS
                self._init_level_parameters()
                reward += 10.0  # Level complete reward
                # SFX: Level Up
                self._create_text_particle(f"LEVEL {self.level}", pygame.Vector2(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            else:
                self.game_over = True # Win condition
                reward += 100.0 # Win game reward

        # --- Check for termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over: # Lost all birds
            reward = -100.0
            self.game_over = True
        
        self.score += reward
        
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if truncated:
            terminated = True # Per Gymnasium API, if truncated is true, terminated should also be true

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, _, _ = action
        if not self.birds:
            return
            
        leader = self.birds[0]
        accel = pygame.Vector2(0, 0)

        if movement == 1: accel.y = -self.BIRD_ACCELERATION  # Up
        elif movement == 2: accel.y = self.BIRD_ACCELERATION   # Down
        elif movement == 3: accel.x = -self.BIRD_ACCELERATION  # Left
        elif movement == 4: accel.x = self.BIRD_ACCELERATION   # Right
        
        leader["vel"] += accel

    def _update_flock(self):
        if not self.birds:
            return

        leader = self.birds[0]
        
        # Apply friction and speed limit to leader
        if leader["vel"].length() > 0:
            leader["vel"] *= self.BIRD_FRICTION
        
        max_speed = self.BIRD_MAX_SPEED * self.flock_speed_multiplier
        if leader["vel"].length() > max_speed:
            leader["vel"].scale_to_length(max_speed)
        
        leader["pos"] += leader["vel"]

        # Keep leader on screen
        leader["pos"].x = np.clip(leader["pos"].x, self.BIRD_SIZE, self.SCREEN_WIDTH - self.BIRD_SIZE)
        leader["pos"].y = np.clip(leader["pos"].y, self.BIRD_SIZE, self.SCREEN_HEIGHT - self.BIRD_SIZE)

        # Update followers
        for i in range(1, len(self.birds)):
            bird = self.birds[i]
            
            # Target position in a V-formation behind the leader
            side = (i % 2 * 2 - 1)
            rank = math.ceil(i/2)
            target_offset = pygame.Vector2(-self.FLOCK_FORMATION_DISTANCE * rank, side * self.FLOCK_FORMATION_DISTANCE * rank * 0.5)
            
            # Rotate offset based on leader's velocity
            if leader["vel"].length() > 0.1:
                angle = leader["vel"].angle_to(pygame.Vector2(1, 0))
                target_offset.rotate_ip(-angle)

            target_pos = leader["pos"] + target_offset
            
            # Move towards target position
            direction_to_target = (target_pos - bird["pos"])
            bird["vel"] += direction_to_target * self.FLOCK_COHESION_STRENGTH
            
            # Apply friction and speed limit to followers
            bird["vel"] *= self.BIRD_FRICTION
            if bird["vel"].length() > max_speed * 1.2: # Followers can be slightly faster to catch up
                bird["vel"].scale_to_length(max_speed * 1.2)
            
            bird["pos"] += bird["vel"]

    def _update_obstacles(self):
        for obstacle in self.obstacles[:]:
            obstacle["pos"].y += self.obstacle_speed
            if obstacle["pos"].y > self.SCREEN_HEIGHT:
                self.obstacles.remove(obstacle)

    def _update_pellets_and_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        if not self.birds:
            return reward

        for bird in self.birds[:]:
            bird_rect = pygame.Rect(bird["pos"].x - self.BIRD_SIZE/2, bird["pos"].y - self.BIRD_SIZE/2, self.BIRD_SIZE, self.BIRD_SIZE)
            
            # Bird-Obstacle collision
            for obstacle in self.obstacles:
                if bird_rect.colliderect(obstacle["rect"]):
                    self.birds.remove(bird)
                    # SFX: Bird hit
                    self._create_explosion(bird["pos"], self.COLOR_OBSTACLE, 15)
                    break # A bird can only hit one obstacle per frame
            else: # continue if the loop wasn't broken
                # Bird-Pellet collision
                for pellet in self.pellets[:]:
                    if bird_rect.colliderect(pellet["rect"]):
                        self.pellets.remove(pellet)
                        reward += 1.0
                        self.total_pellets_collected += 1
                        self.flock_speed_multiplier += 0.02
                        # SFX: Pellet collect
                        self._create_ring_effect(pellet["pos"], self.COLOR_PELLET)
                        break
        return reward

    def _spawn_entities(self):
        # Spawn obstacles
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            self.obstacle_spawn_timer = self.obstacle_spawn_interval
            width = self.np_random.integers(40, 120)
            height = self.np_random.integers(15, 30)
            x = self.np_random.uniform(0, self.SCREEN_WIDTH - width)
            y = -height
            self.obstacles.append({"pos": pygame.Vector2(x, y), "width": width, "height": height, "rect": pygame.Rect(x,y,width,height)})
        
        # Update obstacle rects
        for obs in self.obstacles:
            obs["rect"].topleft = obs["pos"]

        # Spawn pellets
        self.pellet_spawn_timer -= 1
        if self.pellet_spawn_timer <= 0 and len(self.pellets) < 10:
            self.pellet_spawn_timer = self.pellet_spawn_interval
            size = 12
            x = self.np_random.uniform(size, self.SCREEN_WIDTH - size)
            y = self.np_random.uniform(size, self.SCREEN_HEIGHT - size)
            pos = pygame.Vector2(x, y)
            self.pellets.append({"pos": pos, "rect": pygame.Rect(x-size/2, y-size/2, size, size)})

    def _check_termination(self):
        return len(self.birds) == 0 or self.game_over

    # --- Rendering ---

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            # Simple linear interpolation for gradient
            interp = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game_elements(self):
        # Render particles (behind other elements)
        for p in self.particles:
            if p["type"] == "ring":
                alpha = int(255 * (p["life"] / p["max_life"]))
                pygame.gfxdraw.aacircle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), (*p["color"], alpha))
                p["radius"] += 1
            elif p["type"] == "text":
                alpha = int(255 * (p["life"] / p["max_life"]))
                if alpha > 0:
                    text_surf = self.font_large.render(p["text"], True, (*self.COLOR_TEXT, alpha))
                    text_surf.set_alpha(alpha)
                    text_rect = text_surf.get_rect(center=p["pos"])
                    self.screen.blit(text_surf, text_rect)
            else: # square particle
                alpha = 255 * (p["life"] / p["max_life"])
                if alpha > 0:
                    s = pygame.Surface((p["size"], p["size"]), pygame.SRCALPHA)
                    s.fill((*p["color"], int(alpha)))
                    self.screen.blit(s, (p["pos"].x - p["size"]/2, p["pos"].y - p["size"]/2))

        # Render obstacles
        for obs in self.obstacles:
            glow_rect = obs["rect"].inflate(8, 8)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, self.COLOR_OBSTACLE_GLOW, s.get_rect(), border_radius=5)
            self.screen.blit(s, glow_rect.topleft)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs["rect"], border_radius=3)

        # Render pellets
        for pellet in self.pellets:
            pos = (int(pellet["pos"].x), int(pellet["pos"].y))
            radius = int(pellet["rect"].width / 2)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 3, self.COLOR_PELLET_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PELLET)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_PELLET)

        # Render flock
        for i, bird in enumerate(self.birds):
            pos = bird["pos"]
            # Flap animation
            flap = math.sin(self.steps * self.BIRD_FLAP_SPEED + bird["flap_offset"]) * self.BIRD_FLAP_AMP
            
            # Determine orientation from velocity
            if bird["vel"].length() > 0.1:
                angle = bird["vel"].angle_to(pygame.Vector2(1, 0))
            else:
                angle = 0
            
            # Define bird triangle points
            p1 = pygame.Vector2(self.BIRD_SIZE, 0)
            p2 = pygame.Vector2(-self.BIRD_SIZE/2, self.BIRD_SIZE/2 - flap)
            p3 = pygame.Vector2(-self.BIRD_SIZE/2, -self.BIRD_SIZE/2 - flap)

            # Rotate points
            p1.rotate_ip(-angle)
            p2.rotate_ip(-angle)
            p3.rotate_ip(-angle)
            
            # Translate points to bird's position
            points = [p1 + pos, p2 + pos, p3 + pos]
            int_points = [(int(p.x), int(p.y)) for p in points]

            # Draw glow
            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_BIRD_GLOW)
            # Draw bird
            pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_BIRD)
            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_BIRD)
    
    def _render_ui(self):
        def draw_text(text, font, pos, color, shadow_color):
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Level
        draw_text(f"Level: {self.level}/{self.MAX_LEVEL}", self.font_medium, (10, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Birds Remaining
        birds_text = f"Birds: {len(self.birds)}"
        text_w = self.font_medium.size(birds_text)[0]
        draw_text(birds_text, self.font_medium, (self.SCREEN_WIDTH - text_w - 10, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Timer
        timer_text = f"Time: {int(self.level_timer / self.FPS) + 1}"
        text_w = self.font_large.size(timer_text)[0]
        draw_text(timer_text, self.font_large, (self.SCREEN_WIDTH/2 - text_w/2, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Pellets Collected
        pellet_text = f"Pellets: {self.total_pellets_collected}"
        text_w = self.font_small.size(pellet_text)[0]
        draw_text(pellet_text, self.font_small, (self.SCREEN_WIDTH/2 - text_w/2, self.SCREEN_HEIGHT - 30), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "birds_left": len(self.birds),
            "pellets_collected": self.total_pellets_collected,
        }

    def close(self):
        pygame.quit()

    # --- Visual Effects ---

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            vel = pygame.Vector2(self.np_random.uniform(-4, 4), self.np_random.uniform(-4, 4))
            self.particles.append({
                "pos": pygame.Vector2(pos), "vel": vel, "life": self.np_random.integers(10, 20), "max_life": 20,
                "color": color, "size": self.np_random.integers(2, 5), "type": "square"
            })

    def _create_ring_effect(self, pos, color):
        self.particles.append({
            "pos": pygame.Vector2(pos), "vel": pygame.Vector2(0,0), "life": 15, "max_life": 15,
            "color": color, "radius": 5, "type": "ring"
        })
    
    def _create_text_particle(self, text, pos):
        self.particles.append({
            "pos": pygame.Vector2(pos), "vel": pygame.Vector2(0,-1), "life": 45, "max_life": 45,
            "text": text, "type": "text"
        })

    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is not used by the evaluation environment.
    os.environ.setdefault("SDL_VIDEODRIVER", "x11") # Use a visible driver
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Flock Environment")
    clock = pygame.time.Clock()

    total_reward = 0
    
    while not done:
        # --- Human Input ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already the rendered image
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()