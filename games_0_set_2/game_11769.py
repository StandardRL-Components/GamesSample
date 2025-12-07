import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:34:00.944161
# Source Brief: brief_01769.md
# Brief Index: 1769
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Cleanse a corrupted garden by matching colorful flowers to weaken encroaching vines. "
        "Summon friendly creatures to help you fight back and reach the source of light."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Press spacebar near same-colored flowers to perform a match. "
        "Hold shift to summon a helpful creature when you have enough combo points."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    PLAYER_SPEED = 4
    PLAYER_HEALTH_MAX = 5
    VINE_INITIAL_GROWTH_RATE = 15  # Ticks per segment growth
    VINE_ATTACK_RADIUS = 40
    VINE_ATTACK_DAMAGE = 1
    VINE_ATTACK_COOLDOWN = 60  # Ticks
    FLOWER_MATCH_RADIUS = 80
    CREATURE_SUMMON_COST = 10
    CREATURE_LIFESPAN = 300  # Ticks
    CREATURE_SPEED = 2.5
    CREATURE_DAMAGE_RADIUS = 30
    MAX_STEPS = 2000

    # --- COLORS ---
    COLOR_BG = (20, 15, 30)
    COLOR_PLAYER = (100, 150, 255)
    COLOR_PLAYER_GLOW = (50, 80, 200)
    COLOR_VINE = (200, 40, 60)
    COLOR_VINE_WEAK = (120, 80, 90)
    COLOR_SOURCE = (255, 255, 255)
    COLOR_SOURCE_GLOW = (200, 200, 255)
    FLOWER_COLORS = [
        (255, 180, 0),   # Orange
        (255, 80, 210),  # Pink
        (0, 255, 255),   # Cyan
        (240, 240, 50)   # Yellow
    ]
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH = (100, 220, 100)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        self.player_pos = pygame.Vector2(0, 0)
        self.player_health = 0
        self.vines = []
        self.flowers = []
        self.creatures = []
        self.particles = []
        self.combo_points = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.screen_shake = 0

        self.source_pos = pygame.Vector2(self.WIDTH - 50, self.HEIGHT / 2)
        
        self.prev_space_held = False
        self.prev_shift_held = False

        # self.validate_implementation() # No need to call this in the constructor
        # self.reset() # This is called by the agent loop, not needed here

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(50, self.HEIGHT / 2)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.steps = 0
        self.score = 0
        self.combo_points = 0
        self.game_over = False
        self.screen_shake = 0
        self.vine_growth_rate = self.VINE_INITIAL_GROWTH_RATE

        self.vines = []
        self.flowers = []
        self.creatures = []
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False

        for _ in range(5):
            self._spawn_vine()
        for _ in range(15):
            self._spawn_flower()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle player actions and collect event rewards ---
        dist_before = self.player_pos.distance_to(self.source_pos)
        self._handle_movement(movement)
        dist_after = self.player_pos.distance_to(self.source_pos)
        
        if dist_after < dist_before:
            reward += 0.1

        if space_held and not self.prev_space_held:
            reward += self._handle_flower_match()
        
        if shift_held and not self.prev_shift_held:
            reward += self._handle_creature_summon()
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update game state ---
        self._update_vines()
        self._update_creatures()
        self._update_particles()
        
        # --- Handle vine proximity and attacks ---
        is_near_vine = False
        for vine in self.vines:
            if vine['segments'] and self.player_pos.distance_to(vine['segments'][-1]) < self.VINE_ATTACK_RADIUS:
                is_near_vine = True
                if vine['attack_cooldown'] <= 0:
                    self.player_health -= self.VINE_ATTACK_DAMAGE
                    vine['attack_cooldown'] = self.VINE_ATTACK_COOLDOWN
                    self.screen_shake = 10
                    self._create_particles(self.player_pos, (255,0,0), 20)
                    # Sound: Player damage taken
        if is_near_vine:
            reward -= 0.5

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 200 == 0:
            self.vine_growth_rate = max(5, self.vine_growth_rate * 0.95)

        # --- Termination Check ---
        terminated = False
        truncated = False
        if self.player_health <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.player_pos.distance_to(self.source_pos) < 30:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
            
        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement):
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1  # Up
        elif movement == 2: move_vec.y = 1   # Down
        elif movement == 3: move_vec.x = -1  # Left
        elif movement == 4: move_vec.x = 1   # Right
        
        if move_vec.length() > 0:
            move_vec.scale_to_length(self.PLAYER_SPEED)
            self.player_pos += move_vec

        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.HEIGHT)

    def _handle_flower_match(self):
        reward = 0
        for i, flower in enumerate(self.flowers):
            if self.player_pos.distance_to(flower['pos']) < 40:
                # Found a flower to try and match
                match_group = self._find_match_group(i)
                if len(match_group) >= 2:
                    # Sound: Successful match
                    reward += 5.0
                    self.combo_points += len(match_group)
                    
                    # Weaken nearby vines and remove matched flowers
                    center_pos = sum([self.flowers[i]['pos'] for i in match_group], pygame.Vector2()) / len(match_group)
                    self._create_particles(center_pos, self.flowers[match_group[0]]['color'], 30, 5)

                    vines_to_damage = []
                    for vine in self.vines:
                        if vine['segments'] and center_pos.distance_to(vine['segments'][-1]) < self.FLOWER_MATCH_RADIUS:
                           vines_to_damage.append(vine)
                    
                    for vine in vines_to_damage:
                        damage = len(match_group)
                        reward += min(len(vine['segments']), damage) * 2.0
                        vine['segments'] = vine['segments'][:-damage]
                        if not vine['segments']:
                            self.vines.remove(vine)
                            self._spawn_vine()

                    # Remove flowers in reverse index order to avoid issues
                    for index in sorted(match_group, reverse=True):
                        del self.flowers[index]
                        self._spawn_flower()
                    
                    self.screen_shake = 5
                    return reward # Only one match per action
        return 0

    def _find_match_group(self, start_index):
        to_visit = [start_index]
        visited = {start_index}
        match_color = self.flowers[start_index]['color']
        
        head = 0
        while head < len(to_visit):
            current_index = to_visit[head]
            head += 1
            current_pos = self.flowers[current_index]['pos']

            for i, other_flower in enumerate(self.flowers):
                if i not in visited and other_flower['color'] == match_color:
                    if current_pos.distance_to(other_flower['pos']) < 40:
                        visited.add(i)
                        to_visit.append(i)
        return list(visited)

    def _handle_creature_summon(self):
        if self.combo_points >= self.CREATURE_SUMMON_COST:
            # Sound: Summon creature
            self.combo_points -= self.CREATURE_SUMMON_COST
            self.creatures.append({
                'pos': self.player_pos.copy(),
                'lifespan': self.CREATURE_LIFESPAN,
                'target': None
            })
            self._create_particles(self.player_pos, (200, 200, 255), 25, 3)
            return 10.0
        return 0.0

    def _update_vines(self):
        for vine in self.vines:
            vine['growth_cooldown'] -= 1
            vine['attack_cooldown'] -= 1
            
            if vine['growth_cooldown'] <= 0 and len(vine['segments']) < 25:
                vine['growth_cooldown'] = self.vine_growth_rate + random.uniform(-2, 2)
                
                # Update direction slightly
                angle_change = random.uniform(-0.5, 0.5)
                vine['direction'].rotate_ip_rad(angle_change)
                
                # Steer away from edges
                last_pos = vine['segments'][-1]
                if last_pos.x < 50: vine['direction'].x = abs(vine['direction'].x)
                if last_pos.x > self.WIDTH - 50: vine['direction'].x = -abs(vine['direction'].x)
                if last_pos.y < 50: vine['direction'].y = abs(vine['direction'].y)
                if last_pos.y > self.HEIGHT - 50: vine['direction'].y = -abs(vine['direction'].y)

                new_segment = last_pos + vine['direction'] * 10
                vine['segments'].append(new_segment)

    def _update_creatures(self):
        for creature in self.creatures[:]:
            creature['lifespan'] -= 1
            if creature['lifespan'] <= 0:
                self.creatures.remove(creature)
                continue

            # Find closest vine head as target
            closest_vine = None
            min_dist = float('inf')
            for vine in self.vines:
                if vine['segments']:
                    dist = creature['pos'].distance_to(vine['segments'][-1])
                    if dist < min_dist:
                        min_dist = dist
                        closest_vine = vine
            
            creature['target'] = closest_vine
            if creature['target']:
                target_pos = creature['target']['segments'][-1]
                direction = (target_pos - creature['pos']).normalize()
                creature['pos'] += direction * self.CREATURE_SPEED

                if creature['pos'].distance_to(target_pos) < self.CREATURE_DAMAGE_RADIUS:
                    # Damage the vine
                    if creature['target']['segments']:
                        creature['target']['segments'].pop()
                        self._create_particles(target_pos, self.COLOR_VINE, 3, 1)
                        if not creature['target']['segments']:
                           self.vines.remove(creature['target'])
                           self._spawn_vine()

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['vel'] *= 0.95 # Damping
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _spawn_vine(self):
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top': pos = pygame.Vector2(random.uniform(0, self.WIDTH), 0)
        elif edge == 'bottom': pos = pygame.Vector2(random.uniform(0, self.WIDTH), self.HEIGHT)
        elif edge == 'left': pos = pygame.Vector2(0, random.uniform(0, self.HEIGHT))
        else: pos = pygame.Vector2(self.WIDTH, random.uniform(0, self.HEIGHT))
        
        direction = (pygame.Vector2(self.WIDTH/2, self.HEIGHT/2) - pos).normalize()
        self.vines.append({
            'segments': [pos],
            'direction': direction,
            'growth_cooldown': self.VINE_INITIAL_GROWTH_RATE,
            'attack_cooldown': 0
        })

    def _spawn_flower(self):
        self.flowers.append({
            'pos': pygame.Vector2(random.uniform(50, self.WIDTH-100), random.uniform(50, self.HEIGHT-50)),
            'color': random.choice(self.FLOWER_COLORS),
            'pulse_offset': random.uniform(0, 2 * math.pi)
        })

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': random.randint(20, 40),
                'color': color,
                'size': random.uniform(2, 5)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        render_offset = pygame.Vector2(0, 0)
        if self.screen_shake > 0:
            self.screen_shake -= 1
            render_offset.x = random.randint(-self.screen_shake, self.screen_shake)
            render_offset.y = random.randint(-self.screen_shake, self.screen_shake)

        self._render_source(render_offset)
        self._render_vines(render_offset)
        self._render_flowers(render_offset)
        self._render_creatures(render_offset)
        self._render_particles(render_offset)
        self._render_player(render_offset)
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_source(self, offset):
        pos = self.source_pos + offset
        pulse = (math.sin(self.steps * 0.05) + 1) / 2
        radius = 20 + pulse * 5
        glow_radius = radius * 2.5
        
        # Draw glow
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_SOURCE_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (int(pos.x - glow_radius), int(pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), self.COLOR_SOURCE)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), self.COLOR_SOURCE)

    def _render_vines(self, offset):
        for vine in self.vines:
            is_weak = False
            if vine['segments'] and self.player_pos.distance_to(vine['segments'][-1]) < self.FLOWER_MATCH_RADIUS:
                is_weak = True
            
            color = self.COLOR_VINE_WEAK if is_weak else self.COLOR_VINE
            
            for i in range(len(vine['segments'])):
                pos = vine['segments'][i] + offset
                radius = int(2 + i * 0.3)
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, color)
                if i > 0:
                    prev_pos = vine['segments'][i-1] + offset
                    pygame.draw.aaline(self.screen, color, (int(prev_pos.x), int(prev_pos.y)), (int(pos.x), int(pos.y)))

    def _render_flowers(self, offset):
        for flower in self.flowers:
            pos = flower['pos'] + offset
            pulse = (math.sin(self.steps * 0.1 + flower['pulse_offset']) + 1) / 2
            radius = int(8 + pulse * 2)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, flower['color'])
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), radius, flower['color'])

    def _render_creatures(self, offset):
        for creature in self.creatures:
            pos = creature['pos'] + offset
            alpha = int(max(0, min(255, creature['lifespan'] * 2)))
            color = (*self.COLOR_PLAYER, alpha)
            
            p1 = pos + pygame.Vector2(0, -10)
            p2 = pos + pygame.Vector2(-8, 8)
            p3 = pos + pygame.Vector2(8, 8)
            
            if alpha > 0:
                pygame.gfxdraw.filled_trigon(self.screen, int(p1.x), int(p1.y), int(p2.x), int(p2.y), int(p3.x), int(p3.y), color)
                pygame.gfxdraw.aatrigon(self.screen, int(p1.x), int(p1.y), int(p2.x), int(p2.y), int(p3.x), int(p3.y), color)

    def _render_particles(self, offset):
        for p in self.particles:
            pos = p['pos'] + offset
            alpha = int(max(0, min(255, p['lifespan'] * 8)))
            color = (*p['color'], alpha)
            if alpha > 0:
                pygame.draw.circle(self.screen, color, (int(pos.x), int(pos.y)), int(p['size']))

    def _render_player(self, offset):
        pos = self.player_pos + offset
        
        # Glow
        glow_radius = 25
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 80), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (int(pos.x - glow_radius), int(pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Body
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 10, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 10, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Combo
        combo_text = self.font_large.render(f"COMBO: {self.combo_points}", True, self.COLOR_UI_TEXT)
        self.screen.blit(combo_text, (self.WIDTH - combo_text.get_width() - 10, 40))
        if self.combo_points >= self.CREATURE_SUMMON_COST:
            ready_text = self.font_small.render("SUMMON READY [SHIFT]", True, self.COLOR_SOURCE)
            self.screen.blit(ready_text, (self.WIDTH - ready_text.get_width() - 10, 70))

        # Health
        for i in range(self.player_health):
            leaf_rect = pygame.Rect(10 + i * 25, 10, 20, 30)
            pygame.draw.ellipse(self.screen, self.COLOR_HEALTH, leaf_rect)
            pygame.draw.line(self.screen, tuple(c*0.7 for c in self.COLOR_HEALTH), leaf_rect.midtop, leaf_rect.midbottom, 2)
            
        if self.game_over:
            win = self.player_pos.distance_to(self.source_pos) < 30
            msg = "GARDEN CLEANSED" if win else "CONSUMED BY CORRUPTION"
            color = self.COLOR_HEALTH if win else self.COLOR_VINE
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            s = pygame.Surface(text_rect.inflate(20,20).size, pygame.SRCALPHA)
            s.fill((0,0,0,150))
            pygame.draw.rect(s, (0,0,0,150), s.get_rect(), border_radius=5)
            self.screen.blit(s, text_rect.inflate(20,20).topleft)

            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "combo_points": self.combo_points,
            "distance_to_source": self.player_pos.distance_to(self.source_pos)
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Example of how to run the environment ---
    # Set the video driver to a real one for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Setup Pygame window for human interaction
    pygame.display.set_caption("Corrupted Garden")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}")
            print("Press 'R' to reset.")

        clock.tick(30) # Run at 30 FPS

    env.close()