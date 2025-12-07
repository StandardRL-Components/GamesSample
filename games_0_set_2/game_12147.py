import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:24:58.740191
# Source Brief: brief_02147.md
# Brief Index: 2147
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a Martian survival game. The player controls a rover,
    teleporting rocks and ice to gather resources, build a habitat, and survive
    against depleting oxygen and environmental hazards.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Control a rover on Mars to gather resources, build a habitat, and survive against depleting oxygen and environmental hazards."
    user_guide = "Use arrow keys (↑↓←→) to move the rover. Press space to teleport resources and shift to build or upgrade your habitat."
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_LEVEL = 350
    
    # Colors
    COLOR_BG = (10, 5, 5)
    COLOR_SKY = (20, 10, 30)
    COLOR_GROUND = (80, 40, 30)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0, 50)
    COLOR_ROCK = (100, 100, 110)
    COLOR_ICE = (150, 200, 255)
    COLOR_ICE_OUTLINE = (220, 240, 255)
    COLOR_OXYGEN_BUBBLE = (100, 255, 100, 150)
    COLOR_TELEPORT_BEAM = (255, 255, 255)
    COLOR_TARGET_RETICLE = (0, 255, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BAR_BG = (50, 50, 50)
    COLOR_OXYGEN_BAR = (0, 200, 50)
    COLOR_OXYGEN_BAR_WARN = (255, 200, 0)
    COLOR_OXYGEN_BAR_DANGER = (220, 0, 0)
    COLOR_HABITAT_BAR = (0, 150, 255)

    # Game parameters
    MAX_STEPS = 1000
    FPS = 30
    
    # Player
    PLAYER_SIZE = 12
    PLAYER_SPEED = 3.0
    PLAYER_MOMENTUM_DECAY = 0.85
    TELEPORT_RANGE = 40
    TELEPORT_RADIUS = 20
    
    # Oxygen
    MAX_OXYGEN = 100.0
    OXYGEN_CONSUMPTION_IDLE = 0.03
    OXYGEN_CONSUMPTION_MOVE = 0.06
    OXYGEN_BUBBLE_VALUE = 15.0
    HABITAT_REGEN_RATE = 0.2
    HABITAT_RADIUS = 80
    
    # Radiation
    RADIATION_INCREASE_INTERVAL = 200
    RADIATION_INCREASE_AMOUNT = 0.1 # Multiplier increases by this amount

    # Entities
    ROCK_GRAVITY = 0.5
    INITIAL_ROCKS = 15
    INITIAL_ICE = 8
    ICE_MELT_RATE = 0.005 # per step
    ICE_MELT_SPAWN_CHANCE = 0.1
    
    # Building
    BASIC_HABITAT_COST = {'rock': 5, 'ice': 2}
    ADVANCED_HABITAT_COST = {'rock': 10, 'ice': 5}

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
        try:
            self.font_small = pygame.font.SysFont('consolas', 16)
            self.font_large = pygame.font.SysFont('consolas', 24)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 30)

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_dir = None
        self.oxygen = None
        self.radiation = None
        self.resources = None
        self.habitat_level = None
        self.habitat_pos = None
        self.rocks = None
        self.ice_blocks = None
        self.oxygen_bubbles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.stars = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.GROUND_LEVEL - self.PLAYER_SIZE], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.player_dir = np.array([1.0, 0.0], dtype=float)
        
        self.oxygen = self.MAX_OXYGEN
        self.radiation = 1.0
        self.resources = {'rock': 0, 'ice': 0}
        
        self.habitat_level = 0
        self.habitat_pos = None
        
        self.rocks = []
        self.ice_blocks = []
        self._spawn_initial_entities()

        self.oxygen_bubbles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.prev_space_held = False
        self.prev_shift_held = False

        if self.stars is None:
            self.stars = [
                (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.GROUND_LEVEL), random.randint(1, 2))
                for _ in range(150)
            ]
        
        return self._get_observation(), self._get_info()

    def _spawn_initial_entities(self):
        for _ in range(self.INITIAL_ROCKS):
            self._spawn_entity(self.rocks, 'rock')
        for _ in range(self.INITIAL_ICE):
            self._spawn_entity(self.ice_blocks, 'ice')
    
    def _spawn_entity(self, entity_list, entity_type):
        size = random.randint(8, 15)
        pos = np.array([
            random.uniform(size, self.SCREEN_WIDTH - size),
            random.uniform(100, self.GROUND_LEVEL - size)
        ], dtype=float)
        
        # Avoid spawning on player
        while np.linalg.norm(pos - self.player_pos) < 50:
            pos = np.array([
                random.uniform(size, self.SCREEN_WIDTH - size),
                random.uniform(100, self.GROUND_LEVEL - size)
            ], dtype=float)

        entity = {'pos': pos, 'size': size}
        if entity_type == 'ice':
            entity['melt_timer'] = 1.0 # Full health
        entity_list.append(entity)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 1.0  # Survival reward

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Actions ---
        is_moving = self._handle_movement(movement)
        if space_held and not self.prev_space_held:
            teleport_reward = self._handle_teleport()
            reward += teleport_reward
        if shift_held and not self.prev_shift_held:
            build_reward = self._handle_build()
            reward += build_reward
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game State ---
        oxygen_consumed = self._update_player(is_moving)
        reward -= 0.1 * oxygen_consumed

        self._update_rocks()
        self._update_ice()
        
        bubble_reward = self._update_bubbles()
        reward += bubble_reward
        
        self._update_particles()
        self._update_game_logic()
        
        # --- Check Termination ---
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.game_over = terminated
        self.score += reward

        truncated = self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement_action):
        move_vec = np.array([0.0, 0.0])
        if movement_action == 1:  # Up
            move_vec[1] = -1.0
        elif movement_action == 2: # Down
            move_vec[1] = 1.0
        elif movement_action == 3: # Left
            move_vec[0] = -1.0
        elif movement_action == 4: # Right
            move_vec[0] = 1.0
        
        if np.any(move_vec):
            self.player_dir = move_vec
            self.player_vel += self.player_dir * self.PLAYER_SPEED
            return True
        return False

    def _handle_teleport(self):
        target_pos = self.player_pos + self.player_dir * self.TELEPORT_RANGE
        
        found_target = None # Will store (list, index, entity)
        min_dist = float('inf')

        # Find the closest entity in both lists
        for r_list in [self.rocks, self.ice_blocks]:
            for i, entity in enumerate(r_list):
                dist = np.linalg.norm(entity['pos'] - target_pos)
                if dist < self.TELEPORT_RADIUS + entity['size'] and dist < min_dist:
                    min_dist = dist
                    found_target = (r_list, i, entity)
        
        if found_target:
            target_list, target_index, target_entity = found_target
            
            # sfx: TeleportZap.wav
            resource_type = 'rock' if target_list is self.rocks else 'ice'
            self.resources[resource_type] += 1
            
            target_list.pop(target_index) # Safe removal by index
            
            self._create_particles(target_entity['pos'], 30, self.COLOR_TELEPORT_BEAM, 2.0, 15)
            self._spawn_entity(target_list, resource_type) # Respawn a new one
            return 2.0 # Reward for collecting a resource
        return 0.0

    def _handle_build(self):
        if self.habitat_level == 0:
            cost = self.BASIC_HABITAT_COST
            if self.resources['rock'] >= cost['rock'] and self.resources['ice'] >= cost['ice']:
                self.resources['rock'] -= cost['rock']
                self.resources['ice'] -= cost['ice']
                self.habitat_level = 1
                self.habitat_pos = self.player_pos.copy()
                # sfx: BuildSuccess.wav
                self._create_particles(self.habitat_pos, 50, self.COLOR_HABITAT_BAR, 3.0, 25)
                return 10.0 # Brief-specified reward
        elif self.habitat_level == 1:
            cost = self.ADVANCED_HABITAT_COST
            if self.resources['rock'] >= cost['rock'] and self.resources['ice'] >= cost['ice']:
                self.resources['rock'] -= cost['rock']
                self.resources['ice'] -= cost['ice']
                self.habitat_level = 2
                # sfx: BuildUpgrade.wav
                self._create_particles(self.habitat_pos, 100, self.COLOR_PLAYER, 4.0, 30)
                return 50.0 # Brief-specified reward
        # sfx: BuildFail.wav
        return 0.0

    def _update_player(self, is_moving):
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_MOMENTUM_DECAY
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.GROUND_LEVEL - self.PLAYER_SIZE)
        
        base_consumption = self.OXYGEN_CONSUMPTION_MOVE if is_moving or np.linalg.norm(self.player_vel) > 0.1 else self.OXYGEN_CONSUMPTION_IDLE
        total_consumption = base_consumption * self.radiation
        
        if self.habitat_level > 0 and np.linalg.norm(self.player_pos - self.habitat_pos) < self.HABITAT_RADIUS:
            self.oxygen += self.HABITAT_REGEN_RATE
        
        self.oxygen -= total_consumption
        self.oxygen = min(self.oxygen, self.MAX_OXYGEN)
        return total_consumption

    def _update_rocks(self):
        for rock in self.rocks:
            rock['pos'][1] += self.ROCK_GRAVITY
            if rock['pos'][1] > self.GROUND_LEVEL - rock['size']:
                rock['pos'][1] = self.GROUND_LEVEL - rock['size']

    def _update_ice(self):
        i = 0
        while i < len(self.ice_blocks):
            ice = self.ice_blocks[i]
            ice['melt_timer'] -= self.ICE_MELT_RATE
            if ice['melt_timer'] <= 0:
                self.ice_blocks.pop(i)
                # sfx: IceShatter.wav
                self._create_particles(ice['pos'], 10, self.COLOR_ICE, 1.0, 10)
                self._spawn_entity(self.ice_blocks, 'ice') # Respawn
            else:
                if random.random() < self.ICE_MELT_SPAWN_CHANCE:
                    # sfx: BubblePop.wav
                    bubble_pos = ice['pos'] + np.random.uniform(-5, 5, 2)
                    self.oxygen_bubbles.append({'pos': bubble_pos, 'life': 150})
                i += 1

    def _update_bubbles(self):
        collected_reward = 0
        i = 0
        while i < len(self.oxygen_bubbles):
            bubble = self.oxygen_bubbles[i]
            bubble['pos'][1] -= 0.5 # Drift up
            bubble['pos'][0] += math.sin(bubble['pos'][1] / 10) * 0.5 # Wobble
            bubble['life'] -= 1
            
            collided = np.linalg.norm(self.player_pos - bubble['pos']) < self.PLAYER_SIZE + 5
            
            if bubble['life'] <= 0 or collided:
                if collided:
                    self.oxygen += self.OXYGEN_BUBBLE_VALUE
                    collected_reward += 0.5 # Brief-specified reward
                    # sfx: CollectOxygen.wav
                self.oxygen_bubbles.pop(i)
            else:
                i += 1
        return collected_reward

    def _update_particles(self):
        i = 0
        while i < len(self.particles):
            p = self.particles[i]
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.pop(i)
            else:
                i += 1

    def _update_game_logic(self):
        if self.steps > 0 and self.steps % self.RADIATION_INCREASE_INTERVAL == 0:
            self.radiation += self.RADIATION_INCREASE_AMOUNT

    def _check_termination(self):
        if self.oxygen <= 0:
            return True, -100.0
        
        for rock in self.rocks:
            if np.linalg.norm(self.player_pos - rock['pos']) < self.PLAYER_SIZE + rock['size']:
                # sfx: PlayerCrushed.wav
                return True, -100.0
        
        if self.steps >= self.MAX_STEPS:
            if self.habitat_level == 2:
                # sfx: Victory.wav
                return True, 100.0 # Win
            else:
                return True, 0.0 # Timeout
                
        return False, 0.0

    def _create_particles(self, pos, count, color, speed, life):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel_mag = random.uniform(0.5, 1.0) * speed
            vel = np.array([math.cos(angle), math.sin(angle)]) * vel_mag
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': random.randint(life // 2, life),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_SKY)
        for x, y, r in self.stars:
            pygame.draw.circle(self.screen, (200,200,255), (x,y), r)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_LEVEL, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_LEVEL))

    def _render_game(self):
        if self.habitat_level > 0:
            self._render_habitat()
            
        for bubble in self.oxygen_bubbles:
            pos = bubble['pos'].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_OXYGEN_BUBBLE)

        for rock in self.rocks:
            pos = rock['pos'].astype(int)
            size = int(rock['size'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_ROCK)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_ROCK)
        
        for ice in self.ice_blocks:
            pos = ice['pos'].astype(int)
            size = int(ice['size'] * ice['melt_timer'])
            if size > 1:
                pygame.draw.rect(self.screen, self.COLOR_ICE, (pos[0]-size, pos[1]-size, size*2, size*2))
                pygame.draw.rect(self.screen, self.COLOR_ICE_OUTLINE, (pos[0]-size, pos[1]-size, size*2, size*2), 1)

        target_pos = (self.player_pos + self.player_dir * self.TELEPORT_RANGE).astype(int)
        pygame.gfxdraw.aacircle(self.screen, target_pos[0], target_pos[1], self.TELEPORT_RADIUS, self.COLOR_TARGET_RETICLE)

        self._render_player()

        for p in self.particles:
            pos = p['pos'].astype(int)
            life_ratio = p['life'] / 20.0
            alpha = max(0, min(255, int(life_ratio * 255)))
            color = p['color']
            color_with_alpha = (*color[:3], alpha)
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.circle(s, color_with_alpha, (1, 1), 1)
            self.screen.blit(s, (pos[0]-1, pos[1]-1), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_player(self):
        pos = self.player_pos.astype(int)
        size = self.PLAYER_SIZE
        
        glow_surf = pygame.Surface((size*4, size*4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (size*2, size*2), size*2)
        self.screen.blit(glow_surf, (pos[0] - size*2, pos[1] - size*2), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_PLAYER)

        dir_end = (self.player_pos + self.player_dir * size).astype(int)
        pygame.draw.line(self.screen, self.COLOR_BG, pos, dir_end, 3)

    def _render_habitat(self):
        pos = self.habitat_pos.astype(int)
        radius = int(self.HABITAT_RADIUS)
        s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, radius, radius, radius, (0, 100, 150, 30))
        self.screen.blit(s, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

        if self.habitat_level >= 1:
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 25, (150,150,150))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 25, (200,200,200))
        if self.habitat_level == 2:
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1]-10, 20, (180, 220, 255, 150))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1]-10, 20, (220, 240, 255))
    
    def _render_ui(self):
        bar_w, bar_h = 150, 20
        ox_ratio = self.oxygen / self.MAX_OXYGEN if self.MAX_OXYGEN > 0 else 0
        ox_color = self.COLOR_OXYGEN_BAR
        if ox_ratio < 0.5: ox_color = self.COLOR_OXYGEN_BAR_WARN
        if ox_ratio < 0.25: ox_color = self.COLOR_OXYGEN_BAR_DANGER
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 10, bar_w, bar_h))
        pygame.draw.rect(self.screen, ox_color, (10, 10, int(bar_w * ox_ratio), bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, 10, bar_w, bar_h), 1)
        ox_text = self.font_small.render("OXYGEN", True, self.COLOR_UI_TEXT)
        self.screen.blit(ox_text, (15, 12))

        res_text = self.font_small.render(f"Rock: {self.resources['rock']}  Ice: {self.resources['ice']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(res_text, (10, 40))
        
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))
        steps_text = self.font_small.render(f"Step: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH // 2 - steps_text.get_width() // 2, 40))

        status_y = 10
        if self.habitat_level == 0:
            msg = "BUILD BASIC HABITAT [SHIFT]"
            cost = self.BASIC_HABITAT_COST
        elif self.habitat_level == 1:
            msg = "BUILD ADVANCED HABITAT [SHIFT]"
            cost = self.ADVANCED_HABITAT_COST
        else:
            msg = "HABITAT COMPLETE"
            cost = None
        
        status_text = self.font_small.render(msg, True, self.COLOR_UI_TEXT)
        self.screen.blit(status_text, (self.SCREEN_WIDTH - status_text.get_width() - 10, status_y))
        
        if cost:
            cost_text = self.font_small.render(f"Cost: {cost['rock']} Rock, {cost['ice']} Ice", True, self.COLOR_UI_TEXT)
            self.screen.blit(cost_text, (self.SCREEN_WIDTH - cost_text.get_width() - 10, status_y + 20))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "oxygen": self.oxygen,
            "radiation": self.radiation,
            "habitat_level": self.habitat_level,
            "resources": self.resources,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # For human play, we need a real display.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.init()
    pygame.font.init()
    
    pygame.display.set_caption("Martian Survivor")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        movement = 0 # None
        space = 0    # Released
        shift = 0    # Released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

        if done:
            print(f"Game Over!")
            print(f"Final Score: {total_reward:.2f}")
            print(f"Info: {info}")

    env.close()