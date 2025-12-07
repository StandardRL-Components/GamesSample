import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:09:15.789078
# Source Brief: brief_02615.md
# Brief Index: 2615
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An ant-themed Gymnasium environment where the agent must collect crumbs,
    build bridges to cross water, and avoid predators to reach a finish line.
    The agent can shrink to become invisible to predators for a limited time.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Guide an ant to the finish line by collecting crumbs to build bridges over water, "
        "while avoiding predators. Use the shrink ability to become temporarily invisible."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to build a bridge over water (costs crumbs). "
        "Press shift to shrink and become invisible to predators."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.FONT_SIZE = 20

        # Colors
        self.COLOR_BG = (25, 60, 35)
        self.COLOR_BG_ACCENT = (35, 80, 45)
        self.COLOR_WATER = (50, 90, 160)
        self.COLOR_FINISH = (255, 220, 50)
        self.COLOR_PLAYER = (0, 0, 0)
        self.COLOR_PLAYER_GLOW = (0, 255, 255)
        self.COLOR_PREDATOR = (200, 40, 40)
        self.COLOR_PREDATOR_GLOW = (255, 100, 100)
        self.COLOR_CRUMB = (180, 130, 80)
        self.COLOR_BRIDGE = (140, 90, 50)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_BG = (0, 0, 0, 128)

        # Player settings
        self.PLAYER_SPEED = 4.0
        self.PLAYER_SIZE_NORMAL = 8
        self.PLAYER_SIZE_SHRUNK = 4

        # Game progression settings
        self.INITIAL_PREDATORS = 1
        self.INITIAL_SHRUNK_DURATION = 150 # steps
        self.INITIAL_BRIDGE_COST = 5

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", self.FONT_SIZE, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.is_shrunk = False
        self.shrunk_timer = 0
        self.crumb_count = 0
        self.crumbs = []
        self.bridges = []
        self.predators = []
        self.water_zones = []
        self.finish_zone = pygame.Rect(0, 0, 0, 0)
        self.successful_bridges = 0
        self.shrunk_duration_max = self.INITIAL_SHRUNK_DURATION
        self.bridge_cost = self.INITIAL_BRIDGE_COST
        self.predator_speed = 1.0
        self.particles = []

        # Action state management
        self.space_was_held = False
        self.shift_was_held = False

        # Pre-render background for performance
        self.background_surface = self._create_background()
    
    def _create_background(self):
        """Creates a static background surface with a grass texture."""
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        bg.fill(self.COLOR_BG)
        for _ in range(3000):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            pygame.draw.circle(bg, self.COLOR_BG_ACCENT, (x, y), random.randint(1, 2))
        return bg

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player
        self.player_pos = pygame.Vector2(50, self.HEIGHT / 2)
        self.is_shrunk = False
        self.shrunk_timer = 0
        
        # Resources & Progression
        self.crumb_count = 0
        self.successful_bridges = 0
        self.shrunk_duration_max = self.INITIAL_SHRUNK_DURATION
        self.bridge_cost = self.INITIAL_BRIDGE_COST
        self.predator_speed = 1.0
        
        # Level Objects
        self.bridges.clear()
        self.predators.clear()
        self.crumbs.clear()
        self.particles.clear()

        # Action state
        self.space_was_held = False
        self.shift_was_held = False
        
        # --- Procedural Level Generation ---
        self.finish_zone = pygame.Rect(self.WIDTH - 40, 0, 40, self.HEIGHT)
        
        # Water zones
        self.water_zones = [
            pygame.Rect(self.WIDTH * 0.3, 0, 60, self.HEIGHT),
            pygame.Rect(self.WIDTH * 0.65, 0, 60, self.HEIGHT)
        ]

        # Crumbs
        for _ in range(15): # Near side
            self._spawn_crumb(pygame.Rect(20, 20, self.WIDTH * 0.3 - 40, self.HEIGHT - 40))
        for _ in range(15): # Middle
            self._spawn_crumb(pygame.Rect(self.WIDTH * 0.3 + 60, 20, self.WIDTH * 0.3, self.HEIGHT - 40))
        for _ in range(10): # Far side
            self._spawn_crumb(pygame.Rect(self.WIDTH * 0.65 + 60, 20, self.WIDTH * 0.3 - 100, self.HEIGHT-40))

        # Predators
        for _ in range(self.INITIAL_PREDATORS):
            self._spawn_predator()

        return self._get_observation(), self._get_info()

    def _spawn_crumb(self, area):
        crumb_rect = pygame.Rect(
            self.np_random.uniform(area.left, area.right - 10),
            self.np_random.uniform(area.top, area.bottom - 10),
            10, 10
        )
        self.crumbs.append(crumb_rect)

    def _spawn_predator(self):
        patrol_horizontal = self.np_random.choice([True, False])
        if patrol_horizontal:
            start_pos = pygame.Vector2(self.np_random.uniform(100, self.WIDTH-100), self.np_random.uniform(20, self.HEIGHT-20))
            patrol_range = self.np_random.uniform(50, 150)
            p_min = start_pos.x - patrol_range / 2
            p_max = start_pos.x + patrol_range / 2
        else:
            start_pos = pygame.Vector2(self.np_random.uniform(20, self.WIDTH-20), self.np_random.uniform(100, self.HEIGHT-100))
            patrol_range = self.np_random.uniform(50, 100)
            p_min = start_pos.y - patrol_range / 2
            p_max = start_pos.y + patrol_range / 2

        self.predators.append({
            "pos": start_pos,
            "size": 12,
            "horizontal": patrol_horizontal,
            "min": p_min,
            "max": p_max,
            "dir": 1
        })
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Handle Actions ---
        self._handle_input(action)

        # --- Update Game Logic ---
        self._update_player_state()
        self._update_predators()
        self._update_particles()
        
        # --- Check Collisions and Events ---
        event_reward = self._check_collisions_and_events()
        reward += event_reward

        # --- Check Termination Conditions ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = False
        if not self.game_over and self.steps >= self.MAX_STEPS:
            terminated = True
        
        # --- Update Score ---
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1 # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1 # Right
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED

        # Boundary checks
        player_size = self.PLAYER_SIZE_NORMAL if not self.is_shrunk else self.PLAYER_SIZE_SHRUNK
        self.player_pos.x = np.clip(self.player_pos.x, player_size, self.WIDTH - player_size)
        self.player_pos.y = np.clip(self.player_pos.y, player_size, self.HEIGHT - player_size)

        # Build Bridge (Space) - on key press
        space_pressed = space_held and not self.space_was_held
        if space_pressed and self.crumb_count >= self.bridge_cost:
            player_rect = self.get_player_rect()
            can_build = False
            for water in self.water_zones:
                if water.colliderect(player_rect):
                    can_build = True
                    break
            
            if can_build:
                # SFX: build_bridge.wav
                self.crumb_count -= self.bridge_cost
                bridge_size = 20 + self.successful_bridges * 2 # Bridge gets bigger
                new_bridge = pygame.Rect(self.player_pos.x - bridge_size/2, self.player_pos.y - bridge_size/2, bridge_size, bridge_size)
                self.bridges.append(new_bridge)
                self.successful_bridges += 1
                
                # Progression
                if self.successful_bridges % 2 == 0:
                    self.bridge_cost = max(1, self.INITIAL_BRIDGE_COST - self.successful_bridges // 2)
                if self.successful_bridges > 0:
                    self.shrunk_duration_max += 10
                    self._spawn_predator()
                # reward = 1.0 # Reward is handled in _check_collisions_and_events
        
        # Toggle Shrink (Shift) - on key press
        shift_pressed = shift_held and not self.shift_was_held
        if shift_pressed:
            self.is_shrunk = not self.is_shrunk
            if self.is_shrunk:
                # SFX: shrink.wav
                self.shrunk_timer = self.shrunk_duration_max
                self._create_particles(self.player_pos, self.COLOR_PLAYER_GLOW, 20, 2)
            else:
                # SFX: grow.wav
                self.shrunk_timer = 0
                self._create_particles(self.player_pos, self.COLOR_PLAYER, 20, 2)

        self.space_was_held = space_held
        self.shift_was_held = shift_held

    def _update_player_state(self):
        if self.is_shrunk:
            self.shrunk_timer -= 1
            if self.shrunk_timer <= 0:
                self.is_shrunk = False
                # SFX: grow.wav

    def _update_predators(self):
        # Increase speed over time
        if self.steps > 0 and self.steps % 200 == 0:
            self.predator_speed += 0.05

        for p in self.predators:
            if p["horizontal"]:
                p["pos"].x += p["dir"] * self.predator_speed
                if p["pos"].x > p["max"] or p["pos"].x < p["min"]:
                    p["dir"] *= -1
                    p["pos"].x = np.clip(p["pos"].x, p["min"], p["max"])
            else:
                p["pos"].y += p["dir"] * self.predator_speed
                if p["pos"].y > p["max"] or p["pos"].y < p["min"]:
                    p["dir"] *= -1
                    p["pos"].y = np.clip(p["pos"].y, p["min"], p["max"])

    def get_player_rect(self):
        player_size = self.PLAYER_SIZE_NORMAL if not self.is_shrunk else self.PLAYER_SIZE_SHRUNK
        return pygame.Rect(self.player_pos.x - player_size, self.player_pos.y - player_size, player_size * 2, player_size * 2)

    def _check_collisions_and_events(self):
        reward = 0.0
        player_rect = self.get_player_rect()

        # Collect crumbs
        collected_indices = player_rect.collidelistall(self.crumbs)
        if collected_indices:
            for i in sorted(collected_indices, reverse=True):
                # SFX: collect_crumb.wav
                self._create_particles(self.crumbs[i].center, self.COLOR_CRUMB, 10, 1)
                del self.crumbs[i]
                self.crumb_count += 1
                reward += 0.1
        
        # Check water collision
        in_water = False
        for water in self.water_zones:
            if water.colliderect(player_rect):
                in_water = True
                break
        
        if in_water:
            on_bridge = False
            for bridge in self.bridges:
                if bridge.colliderect(player_rect):
                    on_bridge = True
                    break
            if not on_bridge:
                # Pushed out of water
                self.player_pos -= (self.player_pos - pygame.Vector2(player_rect.center)).normalize() * self.PLAYER_SPEED * 1.1

        # Check predator collision
        if not self.is_shrunk:
            for p in self.predators:
                predator_rect = pygame.Rect(p["pos"].x - p["size"], p["pos"].y - p["size"], p["size"] * 2, p["size"] * 2)
                if predator_rect.colliderect(player_rect):
                    # SFX: player_die.wav
                    self.game_over = True
                    reward = -5.0
                    self._create_particles(self.player_pos, self.COLOR_PREDATOR, 50, 4)
                    break
        
        # Check win condition
        if not self.game_over and self.finish_zone.colliderect(player_rect):
            # SFX: win_level.wav
            self.game_over = True
            reward = 100.0
            self._create_particles(self.player_pos, self.COLOR_FINISH, 100, 3)

        return reward

    def _get_observation(self):
        # --- Main Rendering ---
        self.screen.blit(self.background_surface, (0, 0))
        
        # Water and Finish Line
        for water in self.water_zones:
            pygame.draw.rect(self.screen, self.COLOR_WATER, water)
        pygame.draw.rect(self.screen, self.COLOR_FINISH, self.finish_zone)
        
        # Bridges
        for bridge in self.bridges:
            pygame.draw.rect(self.screen, self.COLOR_BRIDGE, bridge)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in self.COLOR_BRIDGE), bridge, 2)
        
        # Crumbs
        for crumb in self.crumbs:
            pygame.draw.rect(self.screen, self.COLOR_CRUMB, crumb)

        # Predators
        for p in self.predators:
            pos = (int(p["pos"].x), int(p["pos"].y))
            size = p["size"]
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(size * 1.4), self.COLOR_PREDATOR_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(size * 1.4), self.COLOR_PREDATOR_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_PREDATOR)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_PREDATOR)
            
        # Player
        if not self.game_over:
            player_size = self.PLAYER_SIZE_NORMAL if not self.is_shrunk else self.PLAYER_SIZE_SHRUNK
            pos = (int(self.player_pos.x), int(self.player_pos.y))
            
            # Glow effect
            glow_color = self.COLOR_PLAYER_GLOW if self.is_shrunk else tuple(c*0.5 for c in self.COLOR_PLAYER_GLOW)
            glow_size = int(player_size * (2.5 if self.is_shrunk else 1.8))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], glow_size, glow_color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_size, glow_color)
            
            # Ant body
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], player_size, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], player_size, self.COLOR_PLAYER)

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))

        # --- UI Overlay ---
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Crumb count
        crumb_text = f"Crumbs: {self.crumb_count} (Cost: {self.bridge_cost})"
        text_surf = self.font.render(crumb_text, True, self.COLOR_UI_TEXT)
        bg_rect = pygame.Rect(5, 5, text_surf.get_width() + 10, text_surf.get_height() + 4)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, bg_rect, border_radius=5)
        self.screen.blit(text_surf, (10, 7))

        # Timer/Steps
        time_text = f"Steps: {self.steps}/{self.MAX_STEPS}"
        text_surf = self.font.render(time_text, True, self.COLOR_UI_TEXT)
        bg_rect = pygame.Rect(self.WIDTH - text_surf.get_width() - 15, 5, text_surf.get_width() + 10, text_surf.get_height() + 4)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, bg_rect, border_radius=5)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 7))

        # Shrunk Timer
        if self.is_shrunk:
            shrunk_text = f"Shrunk: {self.shrunk_timer}"
            text_surf = self.font.render(shrunk_text, True, self.COLOR_UI_TEXT)
            bg_rect = pygame.Rect(self.WIDTH / 2 - text_surf.get_width()/2 - 5, 5, text_surf.get_width() + 10, text_surf.get_height() + 4)
            pygame.draw.rect(self.screen, self.COLOR_UI_BG, bg_rect, border_radius=5)
            self.screen.blit(text_surf, (self.WIDTH / 2 - text_surf.get_width()/2, 7))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crumb_count": self.crumb_count,
            "is_shrunk": self.is_shrunk,
            "successful_bridges": self.successful_bridges,
        }

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(self.np_random.uniform(-max_speed, max_speed), self.np_random.uniform(-max_speed, max_speed)),
                'radius': self.np_random.uniform(2, 5),
                'lifespan': 20,
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.9 # Damping
            p['radius'] *= 0.95
            p['lifespan'] -= 1
            if p['lifespan'] <= 0 or p['radius'] < 0.5:
                self.particles.remove(p)

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you might need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Ant Race")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("Arrows: Move | Space: Build Bridge | Shift: Shrink/Grow")
    print("----------------------\n")
    
    while not done:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Gym step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.metadata["render_fps"])

    print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()