import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:54:26.637777
# Source Brief: brief_02032.md
# Brief Index: 2032
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from dataclasses import dataclass, field

# Using dataclasses for cleaner object management
@dataclass
class Player:
    pos: pygame.Vector2
    size: float
    target_size: float
    speed: float

@dataclass
class Laser:
    p1: pygame.Vector2
    p2: pygame.Vector2
    is_active: bool
    velocity: pygame.Vector2
    bounds: tuple[int, int]  # For oscillating lasers (e.g., (min_y, max_y))

@dataclass
class Element:
    pos: pygame.Vector2
    size: float
    is_toggled: bool
    linked_laser_indices: list[int]
    is_defense_node: bool = False # If it's part of the main defense grid

@dataclass
class Portal:
    rect: pygame.Rect
    target_pos: pygame.Vector2

@dataclass
class Reactor:
    pos: pygame.Vector2
    size: float
    health: int
    is_vulnerable: bool

@dataclass
class Particle:
    pos: pygame.Vector2
    vel: pygame.Vector2
    life: int
    max_life: int
    color: tuple
    start_radius: float

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Infiltrate a high-tech facility, disable laser defenses, and destroy the central reactor. "
        "Shrink to navigate tight spaces and avoid deadly energy beams."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to interact with objects. "
        "Hold shift to shrink and navigate tight spaces."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- CRITICAL: Action and Observation Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame and Display Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.FONT_UI = pygame.font.Font(None, 24)
        self.FONT_MSG = pygame.font.Font(None, 20)

        # --- Visual Style & Colors ---
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_BG_GRID = (20, 30, 50)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 200)
        self.COLOR_LASER = (255, 20, 50)
        self.COLOR_LASER_INACTIVE = (80, 40, 60)
        self.COLOR_ELEMENT = (255, 200, 0)
        self.COLOR_ELEMENT_TOGGLED = (150, 255, 150)
        self.COLOR_PORTAL = (180, 0, 255)
        self.COLOR_REACTOR = (255, 50, 50)
        self.COLOR_REACTOR_VULNERABLE = (255, 150, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_REWARD_MSG_POS = (180, 255, 180)
        self.COLOR_REWARD_MSG_NEG = (255, 180, 180)

        # --- Game State Containers ---
        self.player: Player = None
        self.lasers: list[Laser] = []
        self.elements: list[Element] = []
        self.portals: list[Portal] = []
        self.reactor: Reactor = None
        self.particles: list[Particle] = []

        # --- Game Logic Variables ---
        self.steps = 0
        self.score = 0
        self.last_reward_msg = ""
        self.last_reward_color = self.COLOR_UI_TEXT
        self.last_reward_timer = 0

        # Don't call reset() here, it will be called by the wrapper
        # self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.last_reward_msg = ""
        self.last_reward_timer = 0
        self._setup_level()
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        self.particles.clear()
        self.player = Player(pos=pygame.Vector2(60, self.HEIGHT / 2), size=12.0, target_size=12.0, speed=3.5)
        self.reactor = Reactor(pos=pygame.Vector2(self.WIDTH - 70, self.HEIGHT / 2), size=25.0, health=3, is_vulnerable=False)

        self.elements = [
            Element(pos=pygame.Vector2(200, 100), size=10, is_toggled=False, linked_laser_indices=[0], is_defense_node=True),
            Element(pos=pygame.Vector2(440, 300), size=10, is_toggled=False, linked_laser_indices=[1], is_defense_node=True),
        ]
        self.lasers = [
            Laser(p1=pygame.Vector2(250, 50), p2=pygame.Vector2(250, 350), is_active=True, velocity=pygame.Vector2(0, 0), bounds=(0,0)),
            Laser(p1=pygame.Vector2(350, 100), p2=pygame.Vector2(550, 100), is_active=True, velocity=pygame.Vector2(0, 1.5), bounds=(100, 250)),
        ]
        self.portals = [
            Portal(rect=pygame.Rect(20, 20, 30, 40), target_pos=pygame.Vector2(self.WIDTH - 40, self.HEIGHT - 40)),
            Portal(rect=pygame.Rect(self.WIDTH - 60, self.HEIGHT - 60, 30, 40), target_pos=pygame.Vector2(40, 40)),
        ]

    def step(self, action):
        movement, space_press, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.01  # Small survival reward
        terminated = False
        truncated = False

        # 1. Handle player actions and intentions
        self._handle_resizing(shift_held)
        action_reward = self._handle_actions(space_press)
        reward += action_reward
        self._handle_movement(movement)

        # 2. Update game world state
        self._update_game_state()

        # 3. Check for collisions and terminal conditions
        collision_reward, terminated = self._check_collisions()
        reward += collision_reward

        if self.reactor.health <= 0 and not terminated:
            terminated = True
            reward += 100.0
            self._show_reward_message("REACTOR DESTROYED!", 100)
        
        if self.steps >= 5000:
            truncated = True

        self.score += reward
        self.steps += 1
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_resizing(self, shift_held):
        self.player.target_size = 7.0 if shift_held else 12.0

    def _handle_movement(self, movement):
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length_squared() > 0:
            move_vec.normalize_ip()
            self.player.pos += move_vec * self.player.speed

        # Boundary checks
        self.player.pos.x = np.clip(self.player.pos.x, self.player.size, self.WIDTH - self.player.size)
        self.player.pos.y = np.clip(self.player.pos.y, self.player.size, self.HEIGHT - self.player.size)

    def _handle_actions(self, space_press):
        if not space_press:
            return 0

        # Action priority: Element -> Reactor -> Portal
        # Check for element manipulation
        for el in self.elements:
            if self.player.pos.distance_to(el.pos) < self.player.size + el.size + 10 and not el.is_toggled:
                el.is_toggled = True
                reward = 1.0
                # If it's a main defense node, give bigger reward
                if el.is_defense_node:
                    reward = 5.0
                    self._show_reward_message("Defense Node Disabled", reward)
                else:
                    self._show_reward_message("Element Manipulated", reward)
                
                # Deactivate linked lasers
                for laser_idx in el.linked_laser_indices:
                    self.lasers[laser_idx].is_active = False
                # SFX: element_activate.wav
                self._spawn_particles(el.pos, 15, self.COLOR_ELEMENT_TOGGLED)
                return reward

        # Check for reactor damage
        if self.reactor.is_vulnerable and self.player.pos.distance_to(self.reactor.pos) < self.player.size + self.reactor.size:
            self.reactor.health -= 1
            # SFX: reactor_hit.wav
            self._spawn_particles(self.reactor.pos, 30, self.COLOR_REACTOR_VULNERABLE, 1.5)
            self._show_reward_message("Reactor Hit!", 10)
            return 10.0

        # Check for portal entry
        for portal in self.portals:
            if portal.rect.collidepoint(self.player.pos):
                self.player.pos = portal.target_pos.copy()
                # SFX: portal_whoosh.wav
                self._spawn_particles(portal.target_pos, 20, self.COLOR_PORTAL, 1.2)
                self._show_reward_message("Teleported", 0.5)
                return 0.5
        
        return 0

    def _update_game_state(self):
        # Smoothly interpolate player size
        self.player.size += (self.player.target_size - self.player.size) * 0.15

        # Update lasers
        laser_speed_multiplier = 1.0 + (self.steps / 200) * 0.05
        for laser in self.lasers:
            if laser.velocity.length_squared() > 0:
                new_pos = laser.p1 + laser.velocity * laser_speed_multiplier
                if not laser.bounds[0] <= new_pos.y <= laser.bounds[1]:
                    laser.velocity.y *= -1
                laser.p1 += laser.velocity * laser_speed_multiplier
                laser.p2 += laser.velocity * laser_speed_multiplier

        # Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.pos += p.vel
            p.life -= 1
            p.vel *= 0.95 # Damping

        # Check if reactor should become vulnerable
        if not self.reactor.is_vulnerable:
            if all(e.is_toggled for e in self.elements if e.is_defense_node):
                self.reactor.is_vulnerable = True
                # SFX: reactor_vulnerable.wav
                self._spawn_particles(self.reactor.pos, 50, self.COLOR_REACTOR_VULNERABLE, 2.0)
                self._show_reward_message("REACTOR VULNERABLE!", 20)
        
        # Update reward message timer
        if self.last_reward_timer > 0:
            self.last_reward_timer -= 1

    def _check_collisions(self):
        # Player vs active lasers
        for laser in self.lasers:
            if laser.is_active:
                dist = self._dist_point_to_segment(self.player.pos, laser.p1, laser.p2)
                if dist < self.player.size:
                    # SFX: player_death.wav
                    self._show_reward_message("Laser Collision", -100)
                    return -100.0, True
        return 0.0, False

    def _get_observation(self):
        if self.player is None:
            self._setup_level()
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_objects()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_BG_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_BG_GRID, (0, i), (self.WIDTH, i), 1)

    def _render_game_objects(self):
        # Portals
        for portal in self.portals:
            alpha = 128 + 100 * math.sin(self.steps * 0.1)
            s = pygame.Surface((portal.rect.width, portal.rect.height), pygame.SRCALPHA)
            s.fill((*self.COLOR_PORTAL, alpha))
            self.screen.blit(s, portal.rect.topleft)

        # Reactor
        reactor_color = self.COLOR_REACTOR_VULNERABLE if self.reactor.is_vulnerable else self.COLOR_REACTOR
        pulse = (1 + math.sin(self.steps * 0.1)) * 4 if self.reactor.is_vulnerable else (1 + math.sin(self.steps * 0.05)) * 2
        self._draw_glowing_circle(self.screen, reactor_color, self.reactor.pos, self.reactor.size + pulse, 15)

        # Elements
        for el in self.elements:
            color = self.COLOR_ELEMENT_TOGGLED if el.is_toggled else self.COLOR_ELEMENT
            pygame.gfxdraw.box(self.screen, (int(el.pos.x - el.size), int(el.pos.y - el.size), int(el.size*2), int(el.size*2)), (*color, 180))
            pygame.gfxdraw.rectangle(self.screen, (int(el.pos.x - el.size), int(el.pos.y - el.size), int(el.size*2), int(el.size*2)), color)

        # Lasers
        for laser in self.lasers:
            color = self.COLOR_LASER if laser.is_active else self.COLOR_LASER_INACTIVE
            pygame.draw.aaline(self.screen, color, laser.p1, laser.p2, 3)

        # Particles
        for p in self.particles:
            life_ratio = p.life / p.max_life
            radius = p.start_radius * life_ratio
            alpha = 255 * life_ratio
            if radius > 0:
                self._draw_glowing_circle(self.screen, p.color, p.pos, radius, 0, alpha)

        # Player
        self._draw_glowing_circle(self.screen, self.COLOR_PLAYER, self.player.pos, self.player.size, 10)

    def _render_ui(self):
        # Score
        score_text = self.FONT_UI.render(f"Score: {self.score:.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        # Player Size
        size_text = self.FONT_UI.render(f"Size: {self.player.size:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(size_text, (10, 10))
        # Reward Message
        if self.last_reward_timer > 0:
            alpha = min(255, int(255 * (self.last_reward_timer / 60)))
            msg_surf = self.FONT_MSG.render(self.last_reward_msg, True, self.last_reward_color)
            msg_surf.set_alpha(alpha)
            self.screen.blit(msg_surf, (self.player.pos.x + 20, self.player.pos.y - 20))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "reactor_health": self.reactor.health}

    def _spawn_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = random.randint(20, 50)
            radius = random.uniform(2, 5)
            self.particles.append(Particle(pos.copy(), vel, life, life, color, radius))

    def _show_reward_message(self, text, value):
        sign = "+" if value >= 0 else ""
        self.last_reward_msg = f"{text} [{sign}{value:.0f}]"
        self.last_reward_color = self.COLOR_REWARD_MSG_POS if value >= 0 else self.COLOR_REWARD_MSG_NEG
        self.last_reward_timer = 90  # frames

    @staticmethod
    def _draw_glowing_circle(surface, color, center, radius, glow_size, alpha_override=None):
        center_int = (int(center.x), int(center.y))
        
        # Draw glow
        if glow_size > 0:
            for i in range(glow_size, 0, -2):
                alpha = int(50 * (1 - i / glow_size))
                if alpha_override is not None: alpha = int(alpha_override * (1 - i/glow_size))
                
                s = pygame.Surface((radius * 2 + i * 2, radius * 2 + i * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, alpha), s.get_rect().center, radius + i)
                surface.blit(s, (center_int[0] - radius - i, center_int[1] - radius - i))

        # Draw main circle
        final_alpha = alpha_override if alpha_override is not None else 255
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], int(radius), (*color, int(final_alpha)))
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], int(radius), (*color, int(final_alpha)))

    @staticmethod
    def _dist_point_to_segment(p: pygame.Vector2, a: pygame.Vector2, b: pygame.Vector2):
        if a == b: return p.distance_to(a)
        l2 = (a - b).length_squared()
        t = max(0, min(1, (p - a).dot(b - a) / l2))
        projection = a + t * (b - a)
        return p.distance_to(projection)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is not part of the required Gymnasium interface
    # To use, you might need to unset the dummy video driver
    # e.g., os.environ.pop("SDL_VIDEODRIVER", None)
    
    # For headless execution, we keep the dummy driver
    env = GameEnv(render_mode="rgb_array")
    
    # Manual validation, similar to the original code's test
    try:
        # Test action space
        assert env.action_space.shape == (3,)
        assert env.action_space.nvec.tolist() == [5, 2, 2]
        # Test observation space
        obs, _ = env.reset()
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8
        # Test step
        test_action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")
    except AssertionError as e:
        print(f"✗ Implementation validation failed: {e}")
        
    # Example of running a few steps
    obs, info = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
    env.close()