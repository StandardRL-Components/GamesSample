import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:25:46.923198
# Source Brief: brief_02409.md
# Brief Index: 2409
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a high-speed, bioluminescent racing game.
    The player navigates a procedurally generated bacterial swarm, using teleports
    to pass through portals and avoid obstacles.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Navigate a high-speed bioluminescent racer through a dangerous bacterial swarm. "
        "Avoid obstacles and use your teleport ability to pass through portals."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to teleport, "
        "allowing you to pass through portals."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_DISTANCE = 10000
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (255, 50, 50, 60)
    COLOR_PORTAL = (50, 150, 255)
    COLOR_PORTAL_GLOW = (50, 150, 255, 70)
    COLOR_UI_TEXT = (220, 220, 240)
    SWARM_COLORS = [
        (20, 15, 40), (30, 20, 60), (40, 25, 80),
        (15, 30, 50), (25, 40, 70)
    ]

    # Player settings
    PLAYER_SPEED = 6.0
    PLAYER_RADIUS = 10

    # Teleport settings
    TELEPORT_DISTANCE = 150
    TELEPORT_COOLDOWN_STEPS = 20
    TELEPORT_ACTIVE_STEPS = 4

    # Game dynamics
    INITIAL_WORLD_SPEED = 2.0
    OBSTACLE_SPAWN_CHANCE = 0.02
    PORTAL_SPAWN_CHANCE = 0.015

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)

        # --- Internal State Attributes ---
        # These are initialized in reset()
        self.steps = None
        self.distance_traveled = None
        self.game_over = None
        self.player_pos = None
        self.world_speed = None
        self.obstacle_spawn_prob = None
        self.swarm_pulse_speed = None
        self.teleport_cooldown = None
        self.teleport_active_timer = None
        self.obstacles = None
        self.portals = None
        self.player_trail = None
        self.swarm_particles = None
        
        # Initialize state for the first time
        # self.reset() is called by the wrapper, but we need swarm particles
        self._generate_swarm()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.steps = 0
        self.distance_traveled = 0
        self.game_over = False

        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH * 0.25, self.SCREEN_HEIGHT / 2)
        
        self.world_speed = self.INITIAL_WORLD_SPEED
        self.obstacle_spawn_prob = self.OBSTACLE_SPAWN_CHANCE
        self.swarm_pulse_speed = 0.02

        self.teleport_cooldown = 0
        self.teleport_active_timer = 0
        
        self.obstacles = []
        self.portals = []
        self.player_trail = deque(maxlen=20)
        
        # Swarm is persistent across resets for visual continuity, but can be regenerated if needed
        if self.swarm_particles is None:
            self._generate_swarm()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Base reward for surviving a step

        # --- Update State from Actions and Time ---
        self._handle_input(action)
        self._update_world_state()
        self._update_difficulty()
        
        # --- Handle Collisions and Rewards ---
        collision_reward, terminated = self._check_collisions()
        reward += collision_reward
        
        # --- Check for Win/Loss Conditions ---
        if terminated:
            self.game_over = True
        elif self.distance_traveled >= self.TARGET_DISTANCE:
            reward += 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            # Using truncated instead of terminated for time limit
            truncated = True
            self.game_over = True
            return self._get_observation(), reward, False, truncated, self._get_info()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # truncated
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Player Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED

        # Clamp player position to screen
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

        # Teleport Action
        if space_held and self.teleport_cooldown == 0:
            # sfx: Teleport_Activate
            self.teleport_cooldown = self.TELEPORT_COOLDOWN_STEPS
            self.teleport_active_timer = self.TELEPORT_ACTIVE_STEPS
            self.distance_traveled += self.TELEPORT_DISTANCE
            
            # Check for portal teleport reward
            for portal in self.portals:
                if self.player_pos.distance_to(portal['pos']) < portal['radius']:
                    # reward is handled in step, just mark portal
                    portal['used'] = True # Mark as used
                    # sfx: Portal_Success
                    break

    def _update_world_state(self):
        # Update timers
        if self.teleport_cooldown > 0: self.teleport_cooldown -= 1
        if self.teleport_active_timer > 0: self.teleport_active_timer -= 1

        # Update distance
        self.distance_traveled += self.world_speed

        # Move obstacles and portals
        teleport_shift = self.TELEPORT_DISTANCE if self.teleport_active_timer == self.TELEPORT_ACTIVE_STEPS - 1 else 0
        
        for entity_list in [self.obstacles, self.portals]:
            for entity in entity_list:
                entity['pos'].x -= self.world_speed + teleport_shift
        
        # Remove off-screen entities
        self.obstacles = [o for o in self.obstacles if o['pos'].x > -o['size']]
        self.portals = [p for p in self.portals if p['pos'].x > -p['radius']]

        # Spawn new entities
        if self.np_random.random() < self.obstacle_spawn_prob:
            self._spawn_obstacle()
        if self.np_random.random() < self.PORTAL_SPAWN_CHANCE:
            self._spawn_portal()
            
        # Update player trail
        self.player_trail.append(self.player_pos.copy())

    def _update_difficulty(self):
        # Increase speed every 250 steps
        if self.steps > 0 and self.steps % 250 == 0:
            self.world_speed += 0.05
        # Increase density and pulse every 500 steps
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_spawn_prob = min(0.1, self.obstacle_spawn_prob + 0.01)
            self.swarm_pulse_speed = min(0.1, self.swarm_pulse_speed + 0.02)

    def _check_collisions(self):
        # Player is invincible during teleport
        if self.teleport_active_timer > 0:
            # Check for portal success during teleport
            for p in self.portals:
                if not p['used'] and self.player_pos.distance_to(p['pos']) < p['radius']:
                    p['used'] = True
                    return 10.0, False # Reward for successful portal use
            return 0, False

        # Collision with screen edges (swarm)
        if not (self.PLAYER_RADIUS < self.player_pos.y < self.SCREEN_HEIGHT - self.PLAYER_RADIUS and \
                self.PLAYER_RADIUS < self.player_pos.x < self.SCREEN_WIDTH - self.PLAYER_RADIUS):
            # sfx: Swarm_Collision
            return -5.0, True

        # Collision with obstacles
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_RADIUS, self.player_pos.y - self.PLAYER_RADIUS, self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['pos'].x, obs['pos'].y, obs['size'], obs['size'])
            if player_rect.colliderect(obs_rect):
                # sfx: Obstacle_Hit
                return -10.0, True
        
        return 0, False

    def _spawn_obstacle(self):
        size = self.np_random.integers(20, 50)
        pos = pygame.Vector2(
            self.SCREEN_WIDTH + size,
            self.np_random.uniform(size, self.SCREEN_HEIGHT - size)
        )
        self.obstacles.append({'pos': pos, 'size': size})

    def _spawn_portal(self):
        radius = self.np_random.integers(30, 50)
        pos = pygame.Vector2(
            self.SCREEN_WIDTH + radius,
            self.np_random.uniform(radius, self.SCREEN_HEIGHT - radius)
        )
        # Ensure no immediate overlap with obstacles
        for obs in self.obstacles:
            if pos.distance_to(obs['pos']) < radius + obs['size']:
                return
        self.portals.append({'pos': pos, 'radius': radius, 'used': False})
    
    def _generate_swarm(self):
        self.swarm_particles = []
        for _ in range(150):
            self.swarm_particles.append({
                'pos': pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH),
                                      self.np_random.uniform(0, self.SCREEN_HEIGHT)),
                'radius': self.np_random.uniform(1, 4),
                'color': random.choice(self.SWARM_COLORS),
                'offset': self.np_random.uniform(0, 2 * math.pi)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.distance_traveled,
            "steps": self.steps,
            "world_speed": self.world_speed
        }

    def _render_game(self):
        self._render_swarm()
        self._render_portals()
        self._render_obstacles()
        self._render_player()

    def _render_swarm(self):
        time_factor = self.steps * self.swarm_pulse_speed
        for p in self.swarm_particles:
            pulse = (math.sin(time_factor + p['offset']) + 1) / 2  # 0 to 1
            radius = int(p['radius'] + pulse * 2)
            if radius > 0:
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), radius, p['color'])

    def _render_portals(self):
        for p in self.portals:
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            radius = int(p['radius'])
            color = self.COLOR_PORTAL if not p['used'] else (100, 100, 120) # Dim if used
            glow_color = self.COLOR_PORTAL_GLOW if not p['used'] else (100, 100, 120, 40)
            
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, glow_color)
            
            # Ring
            for i in range(3):
                if radius - i > 0:
                    pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius - i, color)

    def _render_obstacles(self):
        for o in self.obstacles:
            pos_int = (int(o['pos'].x), int(o['pos'].y))
            size = int(o['size'])
            rect = pygame.Rect(pos_int[0], pos_int[1], size, size)
            
            # Simple rectangle for now, but could be polygons
            # Glow
            glow_rect = rect.inflate(size * 0.5, size * 0.5)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, self.COLOR_OBSTACLE_GLOW, glow_surface.get_rect(), border_radius=3)
            self.screen.blit(glow_surface, glow_rect.topleft)

            # Core shape
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=2)

    def _render_player(self):
        # Teleport flash effect
        if self.teleport_active_timer > 0:
            flash_alpha = 200 * (self.teleport_active_timer / self.TELEPORT_ACTIVE_STEPS)
            flash_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            flash_surf.fill((255, 255, 255, flash_alpha))
            self.screen.blit(flash_surf, (0, 0))

        # Player Trail
        if self.player_trail:
            for i, pos in enumerate(self.player_trail):
                alpha = (i / len(self.player_trail)) * 100
                radius = int(self.PLAYER_RADIUS * (i / len(self.player_trail)) * 0.5)
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, (*self.COLOR_PLAYER, int(alpha)))

        # Player Core
        pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS + 5, self.COLOR_PLAYER_GLOW)
        # Center
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        distance_text = f"DISTANCE: {int(self.distance_traveled):05d} / {self.TARGET_DISTANCE}"
        text_surface = self.font_ui.render(distance_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (10, 10))

        if self.game_over:
            if self.distance_traveled >= self.TARGET_DISTANCE:
                msg = "TRACK COMPLETE"
            else:
                msg = "CONNECTION LOST"
            
            end_text_surf = self.font_game_over.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text_surf, text_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Example ---
    # Unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Bioluminescent Racer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        # --- Human Input Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False
                truncated = False
                print("--- Game Reset ---")

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata["render_fps"])

    print(f"Game Over! Final Distance: {info['score']:.0f}, Total Reward: {total_reward:.2f}")
    env.close()