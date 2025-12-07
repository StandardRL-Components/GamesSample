import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:12:01.436455
# Source Brief: brief_02173.md
# Brief Index: 2173
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
    A Gymnasium environment where a player controls a genetically modified seed,
    racing towards the sun at the top of the screen. The seed must navigate
    procedurally generated obstacles, using fertilizer for boosts and strategic
    cards for temporary enhancements.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a genetically modified seed on a race to the sun. Dodge obstacles, use fertilizer for boosts, and play cards for strategic advantages."
    )
    user_guide = (
        "Controls: ↑ to levitate, ↓ to fall faster, ←→ to move sideways. Press space to jump and shift to play your selected card."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000

    # Colors
    COLOR_BG_TOP = (40, 60, 120)
    COLOR_BG_BOTTOM = (10, 20, 50)
    COLOR_SUN = (255, 255, 150)
    COLOR_SUN_GLOW = (255, 255, 0, 50)
    COLOR_SEED = (100, 255, 100)
    COLOR_SEED_GLOW = (150, 255, 150, 100)
    COLOR_OBSTACLE = (139, 69, 19)
    COLOR_OBSTACLE_OUTLINE = (90, 40, 10)
    COLOR_FERTILIZER_BAR = (100, 200, 50)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_WIND = (200, 200, 255, 70)
    COLOR_BOOST_PARTICLE = (150, 255, 150)
    COLOR_COLLISION_PARTICLE = (255, 50, 50)

    # Physics
    GRAVITY = 0.25
    PLAYER_ACCEL = 0.6
    PLAYER_DRAG = 0.95
    JUMP_IMPULSE = -8.0  # Negative is up
    LEVITATE_THRUST = -0.4
    FAST_FALL_ACCEL = 0.4
    MAX_VEL_Y = 10
    MAX_VEL_X = 6

    # Game Mechanics
    FERTILIZER_MAX = 100
    FERTILIZER_REGEN = 0.2
    FERTILIZER_JUMP_COST = 25
    JUMP_COOLDOWN_STEPS = 10
    CARD_PLAY_COOLDOWN_STEPS = 30
    OBSTACLE_SPAWN_Y = -50
    WIN_Y_TARGET = 5000 # World coordinate for the sun

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("sans-serif", 24)
        self.font_card = pygame.font.SysFont("sans-serif", 16)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.camera_y = 0
        self.fertilizer = 0
        self.jump_cooldown = 0
        self.card_play_cooldown = 0
        self.last_shift_held = False
        self.obstacles = []
        self.particles = []
        self.wind_strength = 0
        self.wind_direction = 1
        self.obstacle_density = 0.02
        self.all_cards = []
        self.card_deck = []
        self.card_hand = []
        self.active_effects = {}
        self.last_obstacle_cleared = None

        self._define_cards()
        
        # In a headless environment, reset might not be called automatically,
        # so we ensure state is initialized.
        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.8)
        self.particles = deque(maxlen=200)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.8)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.camera_y = 0
        
        self.fertilizer = self.FERTILIZER_MAX
        self.jump_cooldown = 0
        self.card_play_cooldown = 0
        self.last_shift_held = False

        self.obstacles = []
        self.particles = deque(maxlen=200) # Use deque for efficient appends/pops
        
        self.wind_strength = 0.1
        self.wind_direction = random.choice([-1, 1])
        self.obstacle_density = 0.02
        
        self._reset_cards()
        self.active_effects = {}
        self.last_obstacle_cleared = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- 1. Handle Input & Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal Movement
        if movement == 3: # Left
            self.player_vel.x -= self.PLAYER_ACCEL
        elif movement == 4: # Right
            self.player_vel.x += self.PLAYER_ACCEL

        # Vertical Movement
        if movement == 1: # Levitate
            self.player_vel.y += self.LEVITATE_THRUST * (1 - self.active_effects.get('heavy_seed', {}).get('strength', 0))
        elif movement == 2: # Fast Fall
            self.player_vel.y += self.FAST_FALL_ACCEL
        
        # Jump (Space)
        if space_held and self.jump_cooldown == 0 and self.fertilizer >= self.FERTILIZER_JUMP_COST:
            # sfx: jump_boost.wav
            jump_power = self.JUMP_IMPULSE * self.active_effects.get('growth_spurt', {}).get('strength', 1.0)
            self.player_vel.y = jump_power
            self.fertilizer -= self.FERTILIZER_JUMP_COST
            self.jump_cooldown = self.JUMP_COOLDOWN_STEPS
            for _ in range(20):
                self._create_particle(self.player_pos, self.COLOR_BOOST_PARTICLE, 2, 5, angle_range=(70, 110))

        # Play Card (Shift) - on rising edge
        if shift_held and not self.last_shift_held and self.card_play_cooldown == 0 and self.card_hand:
            # sfx: card_play.wav
            card = self.card_hand.pop(0)
            self.active_effects[card['effect']] = {'duration': card['duration'], 'strength': card.get('strength', 1.0)}
            self.card_play_cooldown = self.CARD_PLAY_COOLDOWN_STEPS
            # Heuristic reward for card usage
            if card['effect'] == 'wind_shield' and self.wind_strength > 0.3:
                reward += 5.0
            elif card['effect'] == 'growth_spurt' and self.player_vel.y < -1:
                reward += 3.0
            else:
                reward -= 2.0 # Penalty for suboptimal play

        self.last_shift_held = shift_held

        # --- 2. Update Game State ---
        self._update_player_physics()
        self._update_camera()
        self._update_environment()
        
        # --- 3. Check Collisions & Rewards ---
        reward += self._check_collisions()
        reward += self.player_vel.y * -0.01 # Continuous reward for upward velocity

        # --- 4. Check Termination ---
        terminated = False
        truncated = False
        world_y = self.camera_y + self.player_pos.y
        
        if world_y < -self.WIN_Y_TARGET: # Win condition (Note: Y is inverted in world space)
            reward += 100
            terminated = True
            self.game_over = True
        elif self.player_pos.y > self.SCREEN_HEIGHT + 20: # Lose condition
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_player_physics(self):
        # Apply wind
        wind_force = self.wind_strength * self.wind_direction * (1 - self.active_effects.get('wind_shield', {}).get('strength', 0))
        self.player_vel.x += wind_force
        
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        
        # Apply drag
        self.player_vel.x *= self.PLAYER_DRAG
        
        # Clamp velocity
        self.player_vel.x = max(-self.MAX_VEL_X, min(self.MAX_VEL_X, self.player_vel.x))
        self.player_vel.y = max(-self.MAX_VEL_Y * 2, min(self.MAX_VEL_Y, self.player_vel.y))

        # Update position
        self.player_pos += self.player_vel
        
        # Screen boundaries (horizontal)
        if self.player_pos.x < 10:
            self.player_pos.x = 10
            self.player_vel.x *= -0.5
        if self.player_pos.x > self.SCREEN_WIDTH - 10:
            self.player_pos.x = self.SCREEN_WIDTH - 10
            self.player_vel.x *= -0.5

    def _update_camera(self):
        # Keep player in the middle 40% of the screen vertically
        # Camera moves up (camera_y becomes more negative) as player moves up
        if self.player_pos.y < self.SCREEN_HEIGHT * 0.3:
            dy = self.player_pos.y - self.SCREEN_HEIGHT * 0.3
            self.camera_y += dy
            self.player_pos.y -= dy
            for obs in self.obstacles:
                obs.y -= dy
            for p in self.particles:
                p['pos'].y -= dy
        elif self.player_pos.y > self.SCREEN_HEIGHT * 0.7:
            dy = self.player_pos.y - self.SCREEN_HEIGHT * 0.7
            self.camera_y += dy
            self.player_pos.y -= dy
            for obs in self.obstacles:
                obs.y -= dy
            for p in self.particles:
                p['pos'].y -= dy

    def _update_environment(self):
        # Update cooldowns and resources
        if self.jump_cooldown > 0: self.jump_cooldown -= 1
        if self.card_play_cooldown > 0: self.card_play_cooldown -= 1
        self.fertilizer = min(self.FERTILIZER_MAX, self.fertilizer + self.FERTILIZER_REGEN)

        # Update card effects
        for effect in list(self.active_effects.keys()):
            self.active_effects[effect]['duration'] -= 1
            if self.active_effects[effect]['duration'] <= 0:
                del self.active_effects[effect]

        # Draw new cards if hand is not full
        if len(self.card_hand) < 3 and len(self.card_deck) > 0:
            self.card_hand.append(self.card_deck.pop(0))

        # Update difficulty
        if self.steps % 200 == 0 and self.steps > 0:
            self.obstacle_density = min(0.1, self.obstacle_density * 1.01)
            self.wind_strength = min(0.5, self.wind_strength + 0.05)

        # Update particles
        for p in list(self.particles):
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # Spawn/despawn obstacles
        self.obstacles = [obs for obs in self.obstacles if obs.top < self.SCREEN_HEIGHT]
        
        # Check if new obstacles should be spawned based on camera position
        spawn_trigger_y = self.OBSTACLE_SPAWN_Y
        if not self.obstacles or self.obstacles[-1].y > spawn_trigger_y + 100:
             if random.random() < self.obstacle_density * 20: # Higher chance when few obstacles
                self._spawn_obstacle()

    def _check_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x - 8, self.player_pos.y - 8, 16, 16)

        for obs in self.obstacles:
            if player_rect.colliderect(obs):
                # sfx: collision.wav
                reward -= 1.0
                self.player_vel.y = max(self.player_vel.y, 2) # Bounce off
                self.player_vel.x *= 0.8
                for _ in range(10):
                    self._create_particle(self.player_pos, self.COLOR_COLLISION_PARTICLE, 1, 4)
                return reward # Only one collision per frame

            # Check for clearing an obstacle
            if self.last_obstacle_cleared != obs and player_rect.bottom < obs.top:
                if abs(player_rect.centerx - obs.centerx) < obs.width:
                    reward += 1.0
                    self.last_obstacle_cleared = obs

        return reward

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "world_y": -self.camera_y,
            "fertilizer": self.fertilizer,
            "active_effects": list(self.active_effects.keys())
        }

    def _render_background(self):
        # Gradient background
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # Sun
        sun_y = (-self.camera_y / self.WIN_Y_TARGET) * self.SCREEN_HEIGHT * 2 + 50
        sun_pos = (self.SCREEN_WIDTH // 2, int(sun_y))
        pygame.gfxdraw.filled_circle(self.screen, sun_pos[0], sun_pos[1], 40, self.COLOR_SUN_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, sun_pos[0], sun_pos[1], 30, self.COLOR_SUN)
        
        # Parallax stars
        for i in range(50):
            seed = i * 42
            x = (seed * 12345) % self.SCREEN_WIDTH
            y = (seed * 54321 - self.camera_y * 0.1) % self.SCREEN_HEIGHT
            pygame.draw.circle(self.screen, (200,200,255), (int(x), int(y)), 1)

    def _render_game(self):
        # Wind particles
        if self.wind_strength > 0.1:
            for _ in range(3):
                if random.random() < self.wind_strength:
                    y = random.randint(0, self.SCREEN_HEIGHT)
                    x = 0 if self.wind_direction > 0 else self.SCREEN_WIDTH
                    self.particles.appendleft({
                        'pos': pygame.math.Vector2(x, y),
                        'vel': pygame.math.Vector2(self.wind_direction * random.uniform(5, 10), 0),
                        'life': random.randint(60, 120),
                        'color': self.COLOR_WIND,
                        'size': random.randint(15, 30)
                    })
        
        # Render particles
        for p in self.particles:
            if 'size' in p and p['size'] > 10: # Wind particle
                pygame.draw.line(self.screen, p['color'], p['pos'], p['pos'] + p['vel']*0.5, int(p['size']/10))
            else: # Standard particle
                size = int(p['life'] / p['max_life'] * p['size'])
                if size > 0:
                    pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), size)

        # Render obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, obs, width=2, border_radius=3)

        # Render player (seed)
        pos_x, pos_y = int(self.player_pos.x), int(self.player_pos.y)
        pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, 12, self.COLOR_SEED_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, 10, self.COLOR_SEED)
        pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, 10, self.COLOR_SEED)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score:.0f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Progress
        progress = (-self.camera_y / self.WIN_Y_TARGET) * 100
        progress_text = self.font_ui.render(f"Progress: {progress:.1f}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(progress_text, (self.SCREEN_WIDTH - progress_text.get_width() - 10, 10))

        # Fertilizer bar
        bar_width = 150
        bar_height = 20
        fill_width = (self.fertilizer / self.FERTILIZER_MAX) * bar_width
        pygame.draw.rect(self.screen, (50, 50, 50), (10, self.SCREEN_HEIGHT - 30, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_FERTILIZER_BAR, (10, self.SCREEN_HEIGHT - 30, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, self.SCREEN_HEIGHT - 30, bar_width, bar_height), 1)

        # Card hand
        for i, card in enumerate(self.card_hand):
            card_rect = pygame.Rect(self.SCREEN_WIDTH - 110 * (3 - i), self.SCREEN_HEIGHT - 60, 100, 50)
            pygame.draw.rect(self.screen, (80, 80, 150), card_rect, border_radius=5)
            pygame.draw.rect(self.screen, (200, 200, 255), card_rect, width=2, border_radius=5)
            card_name = self.font_card.render(card['name'], True, self.COLOR_UI_TEXT)
            self.screen.blit(card_name, (card_rect.x + 5, card_rect.y + 5))
            if i == 0:
                pygame.draw.rect(self.screen, (255, 255, 0), card_rect, width=3, border_radius=5) # Highlight selected

    def _define_cards(self):
        self.all_cards = [
            {'name': 'Wind Shield', 'duration': 300, 'effect': 'wind_shield', 'strength': 0.9},
            {'name': 'Growth Spurt', 'duration': 150, 'effect': 'growth_spurt', 'strength': 1.25},
            {'name': 'Heavy Seed', 'duration': 200, 'effect': 'heavy_seed', 'strength': 0.5}, # Resists wind but harder to levitate
            {'name': 'Fertilizer Pack', 'duration': 1, 'effect': 'fertilizer_pack'}, # Instant effect handled in play logic
        ]

    def _reset_cards(self):
        self.card_deck = self.all_cards * 3
        random.shuffle(self.card_deck)
        self.card_hand = [self.card_deck.pop(0) for _ in range(3) if self.card_deck]

    def _spawn_obstacle(self):
        width = random.randint(50, 150)
        height = random.randint(20, 40)
        x = random.randint(0, self.SCREEN_WIDTH - width)
        y = self.OBSTACLE_SPAWN_Y - height
        self.obstacles.append(pygame.Rect(x, y, width, height))

    def _create_particle(self, pos, color, size, speed, angle_range=(0, 360)):
        angle = math.radians(random.uniform(angle_range[0], angle_range[1]))
        p_speed = random.uniform(speed * 0.5, speed)
        vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * p_speed
        life = random.randint(20, 40)
        self.particles.appendleft({
            'pos': pos.copy(),
            'vel': vel,
            'life': life,
            'max_life': life,
            'color': color,
            'size': random.randint(size-1, size+1)
        })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Seed Racer")
    clock = pygame.time.Clock()

    terminated = False
    truncated = False
    total_reward = 0
    
    # --- Manual Control Mapping ---
    # ARROWS: Move
    # SPACE: Jump/Boost
    # SHIFT: Play Card
    
    while not terminated and not truncated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()